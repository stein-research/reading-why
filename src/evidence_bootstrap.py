"""Shot-noise interval of the evidence measured on the device (the run with --cal-random): every circuit's recorded
counts, overlaps and calibrations alike, are redrawn binomially (parametric bootstrap, as device_bootstrap.py does
for the answer); each draw refits the block's slope and offset from its calibration circuits and recomputes the
hybrid evidence (exact numerators, measured row sums). The circuits are the ones that ran.

python evidence_bootstrap.py [--tag esnli_Born-ap_b2_s0_e20] [--index 3763] [--draws 500]
"""
import json
import math
import re
import sys

import numpy as np
import torch

from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
tag = opt("--tag", "esnli_Born-ap_b2_s0_e20")
idx, draws = int(opt("--index", 3763)), int(opt("--draws", 500))
# the ledger keys of each pair's two blocks (with --cal-random 32), per model
RUNS = {"esnli_Born-ap_b2_s0_e20": {3763: ["ae3d09c69431cc3d", "26df20031defaaa7"],
                                   3765: ["a1d6be83e81a3267", "4327c6e5c51fae25"]}}[tag]
model, d = load(tag)
power = 2 if model.attn == "born2" else 1
ids = torch.tensor(d["x_test"][idx:idx + 1])
tok = np.where(ids[0].numpy() != 0)[0]
present = list(tok) + ([ids.shape[1]] if model.mean_channel else [])


def readings(keys):
    """{block: ([(h, i, j, part, P0 - P1, shots)], [(ideal, P0 - P1, shots)])} from the ledger."""
    out = {}
    for b, k in enumerate(keys):
        rec = json.load(open(f"ibm_jobs/{k}/record.json"))
        counts = json.load(open(f"ibm_jobs/{k}/counts.json"))
        ov, cal = [], []
        for c, cnt in zip(rec["circuits"], counts):
            n = sum(cnt.values())
            v = (cnt.get("0", 0) - cnt.get("1", 0)) / n
            m = re.match(r"block (\d) head (\d) q(\d+) k(\d+) (Re|Im)$", c["note"])
            if m:
                assert int(m[1]) == b
                ov.append((int(m[2]), int(m[3]), int(m[4]), m[5] == "Im", v, n))
                continue
            m = re.match(r"block (\d) random calibration \d+ ideal ([-+.\d]+)$", c["note"])
            assert m and int(m[1]) == b, c["note"]
            cal.append((float(m[2]), v, n))
        out[b] = (ov, cal)
    return out


def redraw(v, n, rng):
    return v if rng is None else 2 * rng.binomial(n, (1 + np.clip(v, -1, 1)) / 2) / n - 1


def overlap_from(table, rng=None):
    rows, fits = [], []

    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        H, L = q.shape[0], q.shape[1]
        ov, cal = table[block]
        x = np.array([c[0] for c in cal])
        y = np.array([redraw(c[1], c[2], rng) for c in cal])
        slope, off = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
        fits.append((float(slope), float(off)))
        re_, im_ = torch.zeros(1, H, L, L, dtype=torch.float64), torch.zeros(1, H, L, L, dtype=torch.float64)
        for h, i, j, imag, v, n in ov:
            scale = np.linalg.norm(q[h, i]) * np.linalg.norm(k[h, j])
            (im_ if imag else re_)[0, h, i, j] = (redraw(v, n, rng) - off) * scale
        exact = torch.einsum("hie,hje->hij", torch.tensor(q).conj(), torch.tensor(k))
        s_ex = (exact.real ** 2 + exact.imag ** 2) ** power
        s_hw = (re_[0] ** 2 + im_[0] ** 2) ** power / slope ** (2 * power)
        keep = torch.zeros(L, dtype=torch.bool)
        keep[present] = True
        num, den = (s_hw * keep).sum(-1), (s_ex * keep).sum(-1)
        rows.append(torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den)))
        return re_, im_
    return overlap, rows, fits


class HwFrozen(Frozen):
    """Frozen whose recorded attention row sums are scaled to the measured ones (as in hw_attention.py)."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, x):
        if not self.replay and x.dim() == 4:
            x = x * self.rows[self.blk][None, :, :, None].to(x.dtype)
            self.blk += 1
        return super().__call__(x)


def hw_evidence(target, rows):
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
    return (x0 * total).detach()[0, tok].numpy()


table = readings(RUNS[idx])
with torch.no_grad():
    pe, _ = compiled_probs(model, ids)
target = pe.argmax(1)
a = evidence(model, ids, target)[0].detach()[0, tok].numpy()
words = [d["words"][t] for t in ids[0][tok].tolist()]


def one(rng):
    overlap, rows, fits = overlap_from(table, rng)
    with torch.no_grad():
        p = compiled_probs(model, ids, overlap=overlap)[0][0].numpy()
    b = hw_evidence(target, rows)
    return p, b, fits


p_dev, b_dev, fit_dev = one(None)
rng = np.random.default_rng(0)
corr, signs, top, top4, ans, fits, bs = [], [], [], [], [], [], []
order = list(np.argsort(-a)[:4])
for n in range(draws):
    p, b, f = one(rng)
    corr.append(np.corrcoef(a, b)[0, 1])
    signs.append(int((np.sign(a) == np.sign(b)).sum()))
    top.append(int(np.argmax(b) == np.argmax(a)))
    top4.append(int(list(np.argsort(-b)[:4]) == order))
    ans.append(int(p.argmax() == pe.argmax()))
    fits.append(f)
    bs.append(b)
    if n % 50 == 49:
        print(f"{n + 1} draws: corr median {np.median(corr):.4f}", flush=True)
lo, hi = np.percentile(corr, [2.5, 97.5])
res = dict(tag=tag, index=idx, ledger=RUNS[idx], draws=draws, words=words, exact=a.tolist(), device=b_dev.tolist(),
           corr_device=float(np.corrcoef(a, b_dev)[0, 1]), corr_ci95=[float(lo), float(hi)],
           corr_median=float(np.median(corr)), signs_all_share=float(np.mean(np.array(signs) == len(a))),
           top_word_share=float(np.mean(top)), top4_order_share=float(np.mean(top4)),
           answer_kept_share=float(np.mean(ans)), fit_device=fit_dev,
           per_word_ci95=np.percentile(np.array(bs), [2.5, 97.5], axis=0).T.tolist())
print(json.dumps({k: v for k, v in res.items() if k not in ("exact", "device", "words", "per_word_ci95")}, indent=1))
for w, x, y, (l, h) in zip(words, a, b_dev, res["per_word_ci95"]):
    print(f"  {w:8s} exact {x:+.3f}  device {y:+.3f}  95% [{l:+.3f}, {h:+.3f}]")
json.dump(res, open(f"results/{tag}_evidence_bootstrap_i{idx}.json", "w"), indent=1)
