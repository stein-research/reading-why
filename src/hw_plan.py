"""Which test pairs would show the hybrid evidence on ibm_kingston, and at what cost? A Monte Carlo of the device
whose noise is the one the recorded runs measured (hw_recal.py), no circuits simulated and no device used.

python hw_plan.py <tag> [--index 3763,3765] [--max-tokens 8] [--draws 50] [--shots 2000] [--cal 32]

Each overlap circuit reads eta_b (1 + s_row r_i) <q|k> + c + s_abs e_c plus binomial shot noise: eta_b the block's
visibility (one per job), r_i a row's own offset, c a readout offset, e_c a circuit's own error. From the recorded
runs (a straight-line fit of measured against ideal, hw_recal.py): slopes 0.77 - 0.82, offsets +0.004 to +0.012,
residual sd 0.027 against 0.021 from shots alone (s_abs = 0.0165), row offsets about 3.5 %. The slope and offset are
estimated from `--cal` calibration circuits of the same depth (Hadamard tests of random state pairs with known
overlap, as noisy as the overlap circuits); every overlap is corrected by the offset and the row sums divided by the
slope's fourth power, so the calibration's error is part of the draw.
"""
import json
import math
import sys

import numpy as np
import torch

from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
tag = args[0]
draws, shots, ncal = int(opt("--draws", 50)), int(opt("--shots", 2000)), int(opt("--cal", 32))
rowcal = int(opt("--row-cal", 0))           # calibration circuits per row: q_i against known partners, each row's own eta
s_row, s_abs = float(opt("--s-row", 0.035)), float(opt("--s-abs", 0.0165))
offset = "--no-offset" not in args                                           # fit and remove the readout offset
model, d = load(tag)
power = 2 if model.attn == "born2" else 1


class HwFrozen(Frozen):
    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, x):
        if not self.replay and x.dim() == 4:
            x = x * self.rows[self.blk][None, :, :, None].to(x.dtype)
            self.blk += 1
        return super().__call__(x)


def hw_evidence(ids, target, rows):
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    fz = HwFrozen(rows)
    with torch.no_grad():
        f(x0, fz)
    total = torch.zeros_like(x0)
    for t, w in zip(*nodes01(math.ceil(model.degree() / 2))):
        x = (float(t) * x0).requires_grad_(True)
        (g,) = torch.autograd.grad(f(x, fz.start_replay()).gather(1, target[:, None]).sum(), x)
        total += float(w) * g
    return (x0 * total).detach()


def draw(present, rng):
    rows = []

    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        H, L = q.shape[:2]
        nq, nk = np.linalg.norm(q, axis=-1), np.linalg.norm(k, axis=-1)
        ov = np.einsum("hie,hje->hij", q.conj(), k) / np.maximum(nq[:, :, None] * nk[:, None, :], 1e-300)
        eta = rng.uniform(0.77, 0.82)
        c0 = rng.uniform(0.004, 0.013)                                        # readout offset towards 0
        vis = eta * (1 + s_row * rng.standard_normal((H, L, 1)))
        read = lambda x, v: np.clip((1 + v * x + c0 + s_abs * rng.standard_normal(x.shape)) / 2, 0, 1)
        cal_true = rng.uniform(-1, 1, ncal)                                   # known overlaps of random pairs
        cal_vis = eta * (1 + s_row * rng.standard_normal(ncal))
        cal = 2 * rng.binomial(shots, read(cal_true, cal_vis)) / shots - 1
        A = np.vstack([cal_true, np.ones(ncal)]).T if offset else cal_true[:, None]
        fit = np.linalg.lstsq(A, cal, rcond=None)[0]
        eta_hat, c_hat = float(fit[0]), float(fit[1]) if offset else 0.0
        m = [((2 * rng.binomial(shots, read(x, vis)) / shots - 1) - c_hat) * nq[:, :, None] * nk[:, None, :]
             for x in (ov.real, ov.imag)]
        eta_row = np.full((H, L, 1), eta_hat)
        if rowcal:                               # each row's own slope, from rowcal circuits sharing its q_i preparation
            t = rng.uniform(0.5, 1.0, (H, L, rowcal)) * rng.choice([-1, 1], (H, L, rowcal))
            y = 2 * rng.binomial(shots, read(t, vis)) / shots - 1 - c_hat
            eta_row = ((y * t).sum(-1) / (t ** 2).sum(-1))[..., None]
        re, im = torch.tensor(m[0])[None], torch.tensor(m[1])[None]
        s_ex = torch.tensor(np.abs(ov * nq[:, :, None] * nk[:, None, :]) ** (2 * power))
        s_hw = (re[0] ** 2 + im[0] ** 2) ** power / torch.tensor(eta_row) ** (2 * power)
        keep = torch.zeros(L, dtype=torch.bool)
        keep[present] = True
        num, den = (s_hw * keep).sum(-1), (s_ex * keep).sum(-1)
        rows.append(torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den)))
        return re, im
    return overlap, rows


if "--index" in args:
    pick = [int(x) for x in opt("--index", "").split(",")]
else:                                                                        # the shortest test pairs
    ntok = (d["x_test"] != 0).sum(1)
    pick = [int(i) for i in np.where(ntok <= int(opt("--max-tokens", 8)))[0]]
res = []
for idx in pick:
    ids = torch.tensor(d["x_test"][idx:idx + 1])
    tok = np.where(ids[0].numpy() != 0)[0]
    present = list(tok) + ([ids.shape[1]] if model.mean_channel else [])
    with torch.no_grad():
        pe, _ = compiled_probs(model, ids)
    target = pe.argmax(1)
    a = evidence(model, ids, target)[0].detach()[0, tok].numpy()
    rng = np.random.default_rng(idx)
    corr, sign, top = [], [], []
    for _ in range(draws):
        overlap, rows = draw(present, rng)
        with torch.no_grad():
            compiled_probs(model, ids, overlap=overlap)
        b = hw_evidence(ids, target, rows)[0, tok].numpy()
        corr.append(np.corrcoef(a, b)[0, 1])
        sign.append(np.mean(np.sign(a) == np.sign(b)))
        top.append(np.argmax(a) == np.argmax(b))
    slots = len(present)
    r = dict(index=idx, words=[d["words"][t] for t in ids[0][tok].tolist()], gold=d["classes"][int(d["y_test"][idx])],
             answer=d["classes"][int(target)], p=float(pe.max()), tokens=len(tok),
             cancellation=float(np.abs(a).sum() / abs(a.sum())), exact=a.tolist(),
             corr_median=float(np.median(corr)), corr_p10=float(np.percentile(corr, 10)),
             signs=float(np.mean(sign)), top_word=float(np.mean(top)),
             circuits=8 * slots ** 2 + 2 * ncal + 4 * slots * rowcal,
             quantum_s=round(0.54 * (8 * slots ** 2 + 2 * ncal + 4 * slots * rowcal) * shots / 2000))
    res.append(r)
    print(f"{idx} [{r['tokens']} tok, {r['gold']}/{r['answer']} {r['p']:.2f}] {' '.join(r['words'])}: cancel "
          f"{r['cancellation']:.2f}  corr {r['corr_median']:.3f} (p10 {r['corr_p10']:.3f}) signs {r['signs']:.0%} "
          f"top {r['top_word']:.0%}  ~{r['circuits']} circuits, ~{r['quantum_s']} s", flush=True)
json.dump(dict(tag=tag, shots=shots, cal=ncal, s_row=s_row, s_abs=s_abs, offset=offset, draws=draws, pairs=res),
          open(f"results/{tag}_hwplan_{shots}_cal{ncal}_row{rowcal}{'_' + opt('--index', '').replace(',', '-') if '--index' in args else ''}.json", "w"), indent=1)
