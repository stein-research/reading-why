"""BEW against the model's own attributions with free denominators (a colleague's review, 2026-11-05).

BEW is integrated gradients with every denominator held at the input. Here, on the same Born Transformer:
1. free IG: integrated gradients of P_g along the same path (a -> alpha a, alpha from 0 to 1) with every denominator
   recomputed at each point; not a polynomial, so Gauss-Legendre with --nodes points. It adds up to
   P_g(a) - lim_{alpha -> 0} P_g(alpha a) (its gap is reported against that), and that limit is not b_g: the reference
   slot does not scale with alpha, so the free path starts from the model's answer at vanishing amplitudes; its
   distance from b_g is reported too. models.GUARD is set to 1e-300 here: at 1e-30 the guard, not the model, sets the
   free path below alpha ~ 1e-7 (a first version of this script reported that artefact as a completeness gap);
2. free Shapley (pairs of at most --max-tok tokens): the exact Shapley value of the real deletion game, v(S) = P_g with
   only the words of S present (the others deleted, every denominator recomputed), v(empty) = b_g, so the values sum
   to P_g - b_g as BEW does.
Each is compared with BEW per pair: rank correlation, the same top word, and the deletion test of its ranking
(flip@1, the five lowest), on the fixed 1,000 test pairs of the tables (free Shapley: the short ones among them).

python free_denominators.py [--nodes 1024] [--max-tok 10] [--pairs 1000] [--seeds 0 1 2]
(the paper, Appendix B: --nodes 1024 --max-tok 14, about 1.5 h on the laptop GPU; the 50-pair checks at 256 and
4,096 nodes: --pairs 50 --seeds 0 --nodes N)
"""
import itertools
import json
import math
import sys

import numpy as np
import torch

import models
import synth
from evidence import _score_fn, evidence, nodes01
from models import build

models.GUARD = 1e-300                                          # see the docstring

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
seeds = [int(s) for s in args[args.index("--seeds") + 1:]] if "--seeds" in args else [0, 1, 2]
NODES, MAXTOK, NPAIRS = int(opt("--nodes", 1024)), int(opt("--max-tok", 10)), int(opt("--pairs", 1000))
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)      # the tables' 1,000 pairs
IDS = torch.tensor(d["x_test"][pick]).to(dev)
GT = torch.tensor(d["gt_test"][pick]).to(dev)
FREE = lambda v: v


def free_ig(model, ids, target, chunk=128):
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    t, w = nodes01(NODES)
    total = torch.zeros_like(x0)
    for i in range(0, NODES, chunk):
        for tt, ww in zip(t[i:i + chunk], w[i:i + chunk]):
            x = (float(tt) * x0).requires_grad_(True)
            out = f(x, FREE).gather(1, target[:, None]).sum()
            (g,) = torch.autograd.grad(out, x)
            total += float(ww) * g
    with torch.no_grad():
        full = f(x0, FREE).gather(1, target[:, None])[:, 0]
        near0 = f(1e-12 * x0, FREE).gather(1, target[:, None])[:, 0]          # the free path is flat below 1e-2
    return (x0 * total).detach(), full - near0, near0


def free_shapley(model, ids1, target):
    pos = (ids1[0] != 0).nonzero()[:, 0].tolist()
    n = len(pos)
    masks = list(itertools.product((0, 1), repeat=n))
    batch = ids1.repeat(len(masks), 1)
    for r, mk in enumerate(masks):
        for k, keep in zip(pos, mk):
            if not keep:
                batch[r, k] = 0
    with torch.no_grad():                                      # 2^n deletions, in chunks
        p = torch.cat([model(batch[c:c + 256])[:, target] for c in range(0, len(batch), 256)])
    share = float(model.share[target])
    v = {mk: (share if sum(mk) == 0 else float(p[r])) for r, mk in enumerate(masks)}
    phi = np.zeros(n)
    for mk, val in v.items():
        s = sum(mk)
        for i in range(n):
            if mk[i] == 0:
                wgt = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                phi[i] += wgt * (v[mk[:i] + (1,) + mk[i + 1:]] - val)
    return pos, phi


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def deletion(model, ids, target, score):
    """flip@1 and flip@5-lowest of a ranking (score [B, L]; absent tokens ignored)."""
    present = ids != 0
    big = torch.finfo(torch.float64).max
    top = torch.where(present, score, -big).argmax(1)
    low = torch.where(present, -score, -big).argsort(1, descending=True)[:, :5]
    n = present.sum(1)
    with torch.no_grad():
        f1 = (model(ids.clone().scatter_(1, top[:, None], 0)).argmax(1) != target).double()
        f5 = (model(ids.clone().scatter_(1, low, 0)).argmax(1) != target).double()
    return torch.where(n > 1, f1, torch.nan), torch.where(n > 5, f5, torch.nan)


out = {"nodes": NODES, "max_tok": MAXTOK, "seeds": {}}
for seed in seeds:
    tag = f"esnli_Born-ap_b2_s{seed}_e20"
    cfg = json.load(open(f"results/{tag}.json"))
    model = build(cfg["model"], len(d["words"]), len(d["classes"]), blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    model.double().eval()
    rows = {k: [] for k in ("rc", "top", "gap", "start", "f1_bew", "f1_free", "f5_bew", "f5_free")}
    for i in range(0, NPAIRS, 50):
        ids = IDS[i:i + 50]
        with torch.no_grad():
            target = model(ids).argmax(1)
        bew = evidence(model, ids, target)[0].detach()
        fig, expl, start = free_ig(model, ids, target)
        rows["gap"] += (fig.sum(1) - expl).abs().tolist()
        rows["start"] += start.abs().tolist()                 # |lim P_g(alpha a) - b_g|, the free path's start
        for b in range(ids.shape[0]):
            pr = (ids[b] != 0).cpu().numpy()
            a1, a2 = bew[b].cpu().numpy()[pr], fig[b].cpu().numpy()[pr]
            rows["rc"].append(rank_corr(a1, a2))
            rows["top"].append(float(np.argmax(a1) == np.argmax(a2)))
        for name, sc in (("bew", bew), ("free", fig)):
            f1, f5 = deletion(model, ids, target, sc)
            rows[f"f1_{name}"] += f1.tolist()
            rows[f"f5_{name}"] += f5.tolist()
        print(f"{tag} {i + 50}/{NPAIRS}", flush=True)
    s = {k: float(np.nanmedian(v)) if k in ("rc", "gap", "start") else float(np.nanmean(v)) for k, v in rows.items()}
    s["gap_max"] = float(np.nanmax(rows["gap"]))
    s["start_max"] = float(np.nanmax(rows["start"]))
    # free Shapley on the short pairs
    short = [i for i in range(NPAIRS) if 2 <= int((IDS[i] != 0).sum()) <= MAXTOK]
    sh = {k: [] for k in ("rc_bew", "top_bew", "rc_free_ig", "f1_shap")}
    for i in short:
        ids1 = IDS[i:i + 1]
        with torch.no_grad():
            target = int(model(ids1).argmax(1))
        pos, phi = free_shapley(model, ids1, target)
        tgt = torch.tensor([target], device=dev)
        a1 = evidence(model, ids1, tgt)[0].detach()[0].cpu().numpy()[pos]
        a2 = free_ig(model, ids1, tgt)[0][0].cpu().numpy()[pos]
        sh["rc_bew"].append(rank_corr(phi, a1))
        sh["top_bew"].append(float(np.argmax(phi) == np.argmax(a1)))
        sh["rc_free_ig"].append(rank_corr(phi, a2))
        sc = torch.zeros_like(ids1, dtype=torch.float64)
        sc[0, pos] = torch.tensor(phi, device=dev)
        sh["f1_shap"].append(float(deletion(model, ids1, tgt, sc)[0][0]))
    s["shapley"] = {"pairs": len(short), "rank_corr_bew_median": float(np.nanmedian(sh["rc_bew"])),
                    "top_word_bew": float(np.mean(sh["top_bew"])),
                    "rank_corr_free_ig_median": float(np.nanmedian(sh["rc_free_ig"])),
                    "flip1_shapley": float(np.nanmean(sh["f1_shap"]))}
    s["shapley_flip1_bew_same_pairs"] = float(np.nanmean([rows["f1_bew"][i] for i in short]))
    out["seeds"][seed] = s
    print(tag, json.dumps(s), flush=True)
    json.dump(out, open(f"results/free_denominators_n{NODES}_p{NPAIRS}.json", "w"), indent=1)     # after every seed
