"""The hybrid evidence of the ibm_kingston runs, recomputed from their recorded counts under other visibility
calibrations (no device used). Which calibration would the evidence on hardware need?

python hw_recal.py <tag> [--index 3763]

The device run of hw_attention.py is replayed from the ledger (ibm_jobs/): block 0's recorded counts give block 0's
overlaps, from which block 1's circuits were built, so block 1's recorded counts belong to the same pass. The
numerators stay exact; the attention row sums are the measured ones, divided by eta_i^4, with eta_i from:
- calib:  the run's own calibration circuits (Hadamard test of q_i with itself; transpiled to 13 CZ, they read ~0.99);
- one:    no correction (eta = 1);
- global: one visibility for the whole block, the slope of measured against ideal over all its overlap circuits
          (what a pilot of deep circuits gives);
- row:    each row's own visibility, the slope over that row's overlap circuits - what a calibration as deep as the
          overlaps would give if it caught each row's loss exactly (an upper bound for depth-matched calibrations);
- row+noise s: row, with a relative error of s drawn per row (a calibration that is depth-matched but imperfect).
"""
import json
import sys

import numpy as np
import torch
from qiskit import qpy
from qiskit.quantum_info import Statevector

from compile_hw import compiled_probs, load
import math

from evidence import _score_fn, evidence, nodes01
from models import Frozen

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
tag = [a for a in args if not a.startswith("--") and a not in (opt("--index", None),)][0]
idx = int(opt("--index", 3763))
JOBS = {3763: ("ibm_jobs/9333cd8540d5532d", "ibm_jobs/e5ebeafc76390413"),
        3765: ("ibm_jobs/76ef8cff395e3d8e", "ibm_jobs/4b394b02a307ff37")}

model, d = load(tag)
ids = torch.tensor(d["x_test"][idx:idx + 1])
present = list(np.where(ids[0].numpy() != 0)[0]) + ([ids.shape[1]] if model.mean_channel else [])
power = 2 if model.attn == "born2" else 1


def recorded(block):
    f = JOBS[idx][block]
    rec = json.load(open(f"{f}/record.json"))
    counts = json.load(open(f"{f}/counts.json"))
    circs = qpy.load(open(f"{f}/circuits.qpy", "rb"))
    return [c["note"] for c in rec["circuits"]], counts, circs, rec["shots"]


def replay(strategy, noise=0.0, seed=0):
    """Overlap callback for compiled_probs: recorded counts; rows receives the measured over exact row sums."""
    rows, rng = [], np.random.default_rng(seed)

    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        notes, counts, circs, shots = recorded(block)
        H, L = q.shape[0], q.shape[1]
        re, im = torch.zeros(1, H, L, L, dtype=torch.float64), torch.zeros(1, H, L, L, dtype=torch.float64)
        eta_cal = np.ones((H, L))
        pts = {}                                                         # (h, i) -> [(ideal, measured)]
        n = 0
        for h in range(H):
            for i in present:
                for j in present:
                    nq, nk = np.linalg.norm(q[h, i]), np.linalg.norm(k[h, j])
                    if nq == 0 or nk == 0:
                        continue
                    for part in (0, 1):
                        assert notes[n] == f"block {block} head {h} q{i} k{j} {('Re', 'Im')[part]}", notes[n]
                        c = counts[n]
                        meas = (c.get("0", 0) - c.get("1", 0)) / shots
                        ov = np.vdot(q[h, i] / nq, k[h, j] / nk)
                        ideal = ov.imag if part else ov.real
                        (im if part else re)[0, h, i, j] = meas * nq * nk
                        pts.setdefault((h, i), []).append((ideal, meas))
                        n += 1
                if np.linalg.norm(q[h, i]) > 0:
                    assert notes[n] == f"block {block} head {h} q{i} k{i} calibration", notes[n]
                    c = counts[n]
                    eta_cal[h, i] = max((c.get("0", 0) - c.get("1", 0)) / shots, 1e-3)
                    n += 1
        assert n == len(notes)
        allp = np.array([p for v in pts.values() for p in v])
        slope = lambda p: float((p[:, 0] * p[:, 1]).sum() / (p[:, 0] ** 2).sum())
        if strategy == "affine":                     # one slope and one readout offset for the block, removed
            fit = np.linalg.lstsq(np.vstack([allp[:, 0], np.ones(len(allp))]).T, allp[:, 1], rcond=None)[0]
            for h in range(H):
                for i in present:
                    for j in present:
                        nn = np.linalg.norm(q[h, i]) * np.linalg.norm(k[h, j])
                        re[0, h, i, j] -= fit[1] * nn
                        im[0, h, i, j] -= fit[1] * nn
        eta = np.ones((H, L))
        for (h, i), p in pts.items():
            eta[h, i] = {"calib": eta_cal[h, i], "one": 1.0, "global": slope(allp),
                         "row": slope(np.array(p)),
                         "affine": fit[0] if strategy == "affine" else 1.0}[strategy] * (1 + noise * rng.standard_normal())
        ov = torch.einsum("hie,hje->hij", torch.tensor(q).conj(), torch.tensor(k))
        s_ex = (ov.real ** 2 + ov.imag ** 2) ** power
        s_hw = (re[0] ** 2 + im[0] ** 2) ** power / torch.tensor(eta)[:, :, None] ** (2 * power)
        keep = torch.zeros(L, dtype=torch.bool)
        keep[present] = True
        num, den = (s_hw * keep).sum(-1), (s_ex * keep).sum(-1)
        r = torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den))
        rows.append(r)
        return re, im
    return overlap, rows


class HwFrozen(Frozen):
    """Frozen whose recorded attention row sums are scaled to the measured ones (as in hw_attention.py)."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, d):
        if not self.replay and d.dim() == 4:
            d = d * self.rows[self.blk][None, :, :, None].to(d.dtype)
            self.blk += 1
        return super().__call__(d)


def hw_evidence_with(model, ids, target, rows):
    """evidence.evidence with the denominators of HwFrozen (hw_attention.hw_evidence)."""
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    fz = HwFrozen(rows)
    with torch.no_grad():
        full = f(x0, fz).gather(1, target[:, None])[:, 0]
        zero = f(torch.zeros_like(x0), fz.start_replay()).gather(1, target[:, None])[:, 0]
    total = torch.zeros_like(x0)
    for t, w in zip(*nodes01(math.ceil(model.degree() / 2))):
        x = (float(t) * x0).requires_grad_(True)
        out = f(x, fz.start_replay()).gather(1, target[:, None]).sum()
        (g,) = torch.autograd.grad(out, x)
        total += float(w) * g
    return (x0 * total).detach(), full, zero


with torch.no_grad():
    pe, _ = compiled_probs(model, ids)
target = pe.argmax(1)
ev_ex, _, _ = evidence(model, ids, target)
tok = np.where(ids[0].numpy() != 0)[0]
a = ev_ex.detach()[0, tok].numpy()
words = [d["words"][t] for t in ids[0][tok].tolist()]
out = dict(tag=tag, index=idx, words=words, exact=a.tolist(),
           cancellation=float(np.abs(a).sum() / abs(a.sum())), runs={})
print(f"{idx} {' '.join(words)}: exact evidence " + ", ".join(f"{w} {x:+.3f}" for w, x in zip(words, a)) +
      f"  (sum|e| / |sum e| = {out['cancellation']:.2f})")
cases = [("calib", 0.0, 0), ("one", 0.0, 0), ("global", 0.0, 0), ("affine", 0.0, 0), ("row", 0.0, 0)]
cases += [("row", s, r) for s in (0.01, 0.03, 0.05) for r in range(20)]
for strategy, noise, seed in cases:
    overlap, rows = replay(strategy, noise, seed)
    with torch.no_grad():
        ph, _ = compiled_probs(model, ids, overlap=overlap)
    b, _, _ = hw_evidence_with(model, ids, target, rows)
    b = b[0, tok].numpy()
    key = strategy if not noise else f"row+{noise}"
    r = dict(corr=float(np.corrcoef(a, b)[0, 1]), top1=bool(np.argmax(a) == np.argmax(b)),
             sign=float(np.mean(np.sign(a) == np.sign(b))), rows=[float(x.min()) for x in rows] +
             [float(x.max()) for x in rows], evidence=b.tolist())
    out["runs"].setdefault(key, []).append(r)
    if not noise:
        lo, hi = min(float(x[:, present].min()) for x in rows), max(float(x[:, present].max()) for x in rows)
        print(f"  {key:7s} rows {lo:.2f}-{hi:.2f}  corr {r['corr']:+.3f}  top word {words[int(np.argmax(b))]}  "
              + ", ".join(f"{w} {y:+.3f}" for w, y in zip(words, b)), flush=True)
for key, v in out["runs"].items():
    if len(v) > 1:
        c = np.array([x["corr"] for x in v])
        print(f"  {key:8s} corr median {np.median(c):+.3f} (min {c.min():+.3f}), top word kept "
              f"{np.mean([x['top1'] for x in v]):.0%}, signs kept {np.mean([x['sign'] for x in v]):.0%}")
json.dump(out, open(f"results/{tag}_recal_i{idx}.json", "w"), indent=1)
