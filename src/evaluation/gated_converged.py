"""The softmax model's gated integrated gradients, converged (referee round 3, M2): as in fairness.py (IG of P_g
over one gate per word, the same 1,000 test pairs), with 4,096 Gauss-Legendre nodes instead of 512. Reports the gap,
the share of pairs whose gap exceeds 0.05, and flip@1, flip@5 lowest and agreement - on all pairs and on the
converged ones - and writes them into results/fairness.json as method 'gated4096' (per pair into results/gated4096.npz).

python gated_converged.py [--nodes 4096]
"""
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import synth
from evidence import nodes01
from models import build

dev = "cuda" if torch.cuda.is_available() else "cpu"
NODES = int(sys.argv[sys.argv.index("--nodes") + 1]) if "--nodes" in sys.argv else 4096
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
IDS = torch.tensor(d["x_test"][pick]).to(dev)
GT = torch.tensor(d["gt_test"][pick]).to(dev)
V, C = len(d["words"]), len(d["classes"])


def gated(model, ids, target, chunk=256):
    x = model.embed(ids).detach()
    f = lambda a: F.softmax(model.from_embedded(x * a[..., None], ids), -1).gather(1, target[:, None])[:, 0]
    one = torch.ones(ids.shape, dtype=x.dtype, device=dev)
    with torch.no_grad():
        expl = f(one) - f(torch.zeros_like(one))
    t, w = nodes01(NODES)
    total = torch.zeros_like(one)
    for i in range(0, NODES, chunk):                     # nodes in chunks: the pairs x nodes batch at once
        tt = torch.tensor(t[i:i + chunk], dtype=x.dtype, device=dev)
        ww = torch.tensor(w[i:i + chunk], dtype=x.dtype, device=dev)
        b = len(tt)
        a = (tt[:, None, None] * one[None]).reshape(-1, ids.shape[1]).requires_grad_(True)
        xx = x.repeat(b, 1, 1)
        ii = ids.repeat(b, 1)
        tg = target.repeat(b)
        out = F.softmax(model.from_embedded(xx * a[..., None], ii), -1).gather(1, tg[:, None])[:, 0].sum()
        (g,) = torch.autograd.grad(out, a)
        total += (ww[:, None, None] * g.view(b, *ids.shape)).sum(0)
    return (one * total).detach(), expl.detach()


def score(prob, ids, target, s, gt):
    present = ids != 0
    ntok = present.sum(1)
    big = torch.finfo(torch.float64).max
    order = torch.where(present, s, -big).argsort(1, descending=True)
    low = torch.where(present, -s, -big).argsort(1, descending=True)
    with torch.no_grad():
        f1 = (prob(ids.clone().scatter_(1, order[:, :1], 0)).argmax(1) != target).double()
        f5 = (prob(ids.clone().scatter_(1, low[:, :5], 0)).argmax(1) != target).double()
    g = gt & present
    k = g.sum(1)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(ids.shape[1], device=dev).expand_as(order))
    agree = ((rank < k[:, None]) & present & g).sum(1).double() / k.clamp(min=1).double()
    return (torch.where(ntok > 1, f1, torch.nan), torch.where(ntok > 5, f5, torch.nan),
            torch.where(k > 0, agree, torch.nan))


per = {}
for seed in range(3):
    tag = f"esnli_Softmax_b2_s{seed}_e20"
    cfg = json.load(open(f"results/{tag}.json"))
    m = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    m.double().eval()
    prob = lambda ids: F.softmax(m(ids), -1)
    rows = {k: [] for k in ("flip1", "flip5_low", "agree", "gap")}
    for i in range(0, 1000, 20):
        ids = IDS[i:i + 20]
        with torch.no_grad():
            t = prob(ids).argmax(1)
        s, expl = gated(m, ids, t)
        gap = (s.sum(1) - expl).abs()
        for k, v in zip(("flip1", "flip5_low", "agree"), score(prob, ids, t, s, GT[i:i + 20])):
            rows[k].append(v.cpu())
        rows["gap"].append(gap.cpu())
    for k in rows:
        per[f"s{seed}_{k}"] = torch.cat(rows[k]).numpy()
    gp = per[f"s{seed}_gap"]
    print(tag, f"gap median {np.median(gp):.2e} max {gp.max():.2e} share > 0.05 {np.mean(gp > 0.05):.3f}", flush=True)
np.savez("results/gated4096.npz", **per)

rng = np.random.default_rng(1)
boot = rng.integers(0, 1000, size=(2000, 1000))
gaps = np.stack([per[f"s{s}_gap"] for s in range(3)])
conv = (gaps <= 0.05).all(0)                                # pairs converged in every seed
out = {"nodes": NODES, "gap_median": float(np.median(gaps)), "gap_max": float(gaps.max()),
       "share_gap_over_0.05": float(np.mean(gaps > 0.05)), "converged_pairs": int(conv.sum())}
for key in ("flip1", "flip5_low", "agree"):
    a = np.nanmean(np.stack([per[f"s{s}_{key}"] for s in range(3)]), 0)
    ok = ~np.isnan(a)
    ms = [np.nanmean(a[b][ok[b]]) for b in boot]
    out[key] = [float(np.nanmean(a[ok])), [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]]
    out[key + "_converged"] = float(np.nanmean(a[ok & conv]))
print(json.dumps(out, indent=1))
fair = json.load(open("results/fairness.json"))
fair["softmax"]["gated4096"] = out
json.dump(fair, open("results/fairness.json", "w"), indent=1)
