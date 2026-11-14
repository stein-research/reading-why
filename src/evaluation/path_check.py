"""The held-denominator path of the Born Transformer (a note from the Born LM work, 2026-11-06): there, F(alpha a) stayed
near 0 until alpha ~ 1 (like alpha^k, k in the hundreds), so BEW ranked tokens as gradient x input on the held path
(IG = gxi / k for alpha^k), and on some models the path left [0, 1] by many orders, so float64 broke completeness.
Here, on the three e-SNLI seeds and the tables' 1,000 test pairs:
- k_eff = sum_t gxi_t / sum_t BEW_t per pair, gxi_t = a_t dF/da_t at the input with the denominators held;
- the rank correlation of BEW and held gxi, the same top word, and flip@1 / flip@5 lowest of both rankings;
- g(alpha) = F(alpha a) / F(a) at alpha = 0.25, 0.5, 0.75, and the path's peak max_alpha |F(alpha a)| over 201
  points against |F(a)|;
- the completeness gap, median and largest.
Writes results/path_check.json.

python path_check.py      (about an hour on the laptop GPU)
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import synth
from evidence import _score_fn, evidence
from models import Frozen, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
IDS = torch.tensor(d["x_test"][pick]).to(dev)
ALPHA = np.linspace(0, 1, 201)


def rank_corr(a, b):                                    # as in free_denominators.py
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def deletion(model, ids, target, score):                # as in free_denominators.py
    present = ids != 0
    big = torch.finfo(torch.float64).max
    top = torch.where(present, score, -big).argmax(1)
    low = torch.where(present, -score, -big).argsort(1, descending=True)[:, :5]
    n = present.sum(1)
    with torch.no_grad():
        f1 = (model(ids.clone().scatter_(1, top[:, None], 0)).argmax(1) != target).double()
        f5 = (model(ids.clone().scatter_(1, low, 0)).argmax(1) != target).double()
    return torch.where(n > 1, f1, torch.nan), torch.where(n > 5, f5, torch.nan)


out = {}
for seed in range(3):
    tag = f"esnli_Born-ap_b2_s{seed}_e20"
    cfg = json.load(open(f"results/{tag}.json"))
    model = build(cfg["model"], len(d["words"]), len(d["classes"]), blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    model.double().eval()
    rows = {k: [] for k in ("k_eff", "rc", "top", "g25", "g50", "g75", "peak", "gap",
                            "f1_bew", "f1_gxi", "f5_bew", "f5_gxi")}
    for i in range(0, 1000, 50):
        ids = IDS[i:i + 50]
        with torch.no_grad():
            target = model(ids).argmax(1)
        bew, full, zero = evidence(model, ids, target)
        bew = bew.detach()
        x0, f = _score_fn(model, ids)
        x0 = x0.detach()
        fz = Frozen()
        with torch.no_grad():
            f(x0, fz)                                             # records the input's denominators
        x = x0.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(f(x, fz.start_replay()).gather(1, target[:, None]).sum(), x)
        gxi = (x0 * g).detach()
        with torch.no_grad():
            path = torch.stack([f(float(a) * x0, fz.start_replay()).gather(1, target[:, None])[:, 0] for a in ALPHA], 1)
        expl = full - zero
        rows["k_eff"] += (gxi.sum(1) / bew.sum(1)).tolist()
        rows["gap"] += (bew.sum(1) - expl).abs().tolist()
        rows["peak"] += (path.abs().amax(1) / full.abs()).tolist()
        for a, k in ((0.25, "g25"), (0.5, "g50"), (0.75, "g75")):
            rows[k] += (path[:, int(round(a * 200))] / full).tolist()
        for b in range(ids.shape[0]):
            pr = (ids[b] != 0).cpu().numpy()
            a1, a2 = bew[b].cpu().numpy()[pr], gxi[b].cpu().numpy()[pr]
            rows["rc"].append(rank_corr(a1, a2))
            rows["top"].append(float(np.argmax(a1) == np.argmax(a2)))
        for name, sc in (("bew", bew), ("gxi", gxi)):
            f1, f5 = deletion(model, ids, target, sc)
            rows[f"f1_{name}"] += f1.tolist()
            rows[f"f5_{name}"] += f5.tolist()
    s = {k: float(np.nanmedian(v)) for k, v in rows.items() if not k.startswith("f") and k != "top"}
    s.update({k: float(np.nanmean(v)) for k, v in rows.items() if k.startswith("f") or k == "top"})
    s["k_eff_p5_p95"] = [float(np.nanpercentile(rows["k_eff"], 5)), float(np.nanpercentile(rows["k_eff"], 95))]
    s["k_eff_negative"] = float(np.mean(np.array(rows["k_eff"]) < 0))
    s["peak_max"] = float(np.nanmax(rows["peak"]))
    s["gap_max"] = float(np.nanmax(rows["gap"]))
    out[seed] = s
    print(tag, json.dumps(s), flush=True)
json.dump(out, open("results/path_check.json", "w"), indent=1)
