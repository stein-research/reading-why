"""What the ibm_kingston runs can show (referee round 5, 2026-11-12): the power of the device checks of Section 5.

(2026-11-12, referee round 5: uniform and random overlaps are now given to the input's present slots only; before,
absent slots took a share of each row - uniform rows summed to 0.12 - and the nulls were wrong.)
Answers. On each device input (pairs 3763, 3765 and 3763 without "aliens"), the compiled Born Transformer (seed 0)
with its attention overlaps replaced by
  uniform   every overlap equal (each attention row uniform), and
  random    independent Gaussian real and imaginary parts (--draws draws);
an answer check can tell the device from uninformative attention only where these answer otherwise.
Evidence. On the two pairs with depth-matched calibrations, BEW with the attention row sums scaled per row (as
hw_attention.HwFrozen does), correlated with the exact BEW, for
  device      the device's measured, visibility-corrected row sums (the paper's 0.962 / 0.997);
  no-vis      the same without the visibility correction (each row sum times its block's slope squared);
  random      unbiased log-normal row errors with the device's own spread (sd of log measured/exact), --draws draws;
  seeds 1, 2  BEW of the independently trained seeds on the same pair.
Error budget. For every device block with random calibrations: the calibrations' scatter about their fitted line
(ddof 2) against the shot noise at the measured values, sqrt(mean((1 - p^2) / shots)).

python device_power.py [--draws 100]   -> results/device_power.json   (a few minutes on the CPU; no device used)
"""
import json
import math
import sys

import numpy as np
import torch

from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 100
TAG = "esnli_Born-ap_b2_s0_e20"
JOBS = {3763: ("ae3d09c69431cc3d", "26df20031defaaa7"), 3765: ("a1d6be83e81a3267", "4327c6e5c51fae25")}
rng = np.random.default_rng(0)
model, d = load(TAG)
C = d["classes"]


PRESENT = None            # [L] bool: the slots of the current input (its tokens and the reference slot)


def _mask(x):             # only present slots: compiled_probs normalises each row before it masks absent ones, so
    m = torch.tensor(PRESENT)                 # an overlap given to an absent slot would take a share of the row
    return x * (m[None, None, :, None] & m[None, None, None, :]).to(x.dtype)


def uniform_ov(qi, kj, block):
    ov = (qi.conj() * kj).sum(-1)
    return _mask(torch.ones_like(ov.real)), torch.zeros_like(ov.real)


def random_ov(qi, kj, block):
    ov = (qi.conj() * kj).sum(-1)
    return _mask(torch.tensor(rng.standard_normal(ov.shape))), _mask(torch.tensor(rng.standard_normal(ov.shape)))


class HwFrozen(Frozen):                       # as hw_attention.HwFrozen (that module runs a pass on import)
    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, dd):
        if not self.replay and dd.dim() == 4:
            dd = dd * self.rows[self.blk][None, :, :, None].to(dd.dtype)
            self.blk += 1
        return super().__call__(dd)


def hw_ev(ids, target, rows):
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    fz = HwFrozen(rows)
    with torch.no_grad():
        f(x0, fz)
    tot = torch.zeros_like(x0)
    for t, w in zip(*nodes01(math.ceil(model.degree() / 2))):
        x = (float(t) * x0).requires_grad_(True)
        (g,) = torch.autograd.grad(f(x, fz.start_replay()).gather(1, target[:, None]).sum(), x)
        tot += float(w) * g
    return (x0 * tot).detach()


out = {"draws": DRAWS, "answers": {}, "evidence": {}, "error_budget": []}
with torch.no_grad():
    for name, idx, dele in (("3763", 3763, None), ("3765", 3765, None), ("3763 without aliens", 3763, 5)):
        ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
        if dele is not None:
            assert d["words"][int(ids[0, dele])] == "aliens"
            ids[0, dele] = 0
        PRESENT = np.concatenate([ids[0].numpy() != 0, [True] * int(model.mean_channel)])
        pe = compiled_probs(model, ids)[0][0]
        pu = compiled_probs(model, ids, overlap=uniform_ov)[0][0]
        pr = torch.stack([compiled_probs(model, ids, overlap=random_ov)[0][0] for _ in range(DRAWS)])
        out["answers"][name] = dict(exact=C[int(pe.argmax())], exact_p=pe.tolist(), uniform=C[int(pu.argmax())],
                                    uniform_p=pu.tolist(),
                                    random_same_answer=float((pr.argmax(-1) == pe.argmax()).double().mean()))
        print(name, out["answers"][name], flush=True)

m1, _ = load("esnli_Born-ap_b2_s1_e20")
m2, _ = load("esnli_Born-ap_b2_s2_e20")
for idx in (3763, 3765):
    r = json.load(open(f"results/{TAG}_device_ibm_kingston_2000_i{idx}_cal32.json"))
    ids = torch.tensor(d["x_test"][idx:idx + 1])
    tok = np.where(ids[0].numpy() != 0)[0]
    with torch.no_grad():
        target = model(ids).argmax(1)
    ex = evidence(model, ids, target)[0][0, tok].detach().numpy()
    corr = lambda e: float(np.corrcoef(ex, e)[0, 1])
    rows = [torch.tensor(np.array(b)) for b in r["device_run"]["eta"]]
    slopes = [c["slope"] for c in r["device_run"]["calibration"]]
    o = dict(device=corr(hw_ev(ids, target, rows)[0, tok].numpy()),
             no_visibility=corr(hw_ev(ids, target, [b * s ** 2 for b, s in zip(rows, slopes)])[0, tok].numpy()))
    sd = float(np.mean([np.log(b[b != 1.0].numpy()).std() for b in rows]))
    cs = np.array([corr(hw_ev(ids, target, [torch.where(b != 1.0, torch.tensor(np.exp(rng.normal(0, sd, b.shape))), b)
                                            for b in rows])[0, tok].numpy()) for _ in range(DRAWS)])
    o.update(row_log_sd=sd, random_median=float(np.median(cs)), random_p5_p95=[float(np.percentile(cs, 5)),
             float(np.percentile(cs, 95))], random_share_below_device=float((cs < o["device"]).mean()))
    for nm, mm in (("seed1", m1), ("seed2", m2)):
        with torch.no_grad():
            t2 = mm(ids).argmax(1)
        o[nm] = corr(evidence(mm, ids, t2)[0][0, tok].detach().numpy())
    out["evidence"][str(idx)] = o
    print(idx, o, flush=True)
    for j in JOBS[idx]:
        rec = json.load(open(f"ibm_jobs/{j}/record.json"))
        cnt = json.load(open(f"ibm_jobs/{j}/counts.json"))
        shots = int(rec["shots"])
        cal = [(float(c["note"].split("ideal ")[1]), (n.get("0", 0) - n.get("1", 0)) / shots)
               for c, n in zip(rec["circuits"], cnt) if "random calibration" in c["note"]]
        x, p = np.array(cal).T
        a, b = np.polyfit(x, p, 1)
        sc = float(np.sqrt(((p - a * x - b) ** 2).sum() / (len(p) - 2)))
        sh = float(np.sqrt(np.mean((1 - p ** 2) / shots)))
        out["error_budget"].append(dict(pair=idx, job=j, slope=float(a), scatter=sc, shot=sh, ratio=sc / sh))
        print("  ", out["error_budget"][-1], flush=True)
json.dump(out, open("results/device_power.json", "w"), indent=1)
