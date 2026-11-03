"""Deletion test, Shapley comparison and (ORDER) ground-truth check for trained runs.

Deletion: on a fixed sample of test queries, explain the predicted class, remove the k tokens of highest evidence
(k = 1..5; only queries with more than k tokens), and measure the drop of the model's own probability of that class
(BHR p-hat for B, softmax for S and P1) and how often the prediction changes. Orders compared: evidence, random,
lowest evidence first. Removing a token is id 0: absent for S and P1, amplitude 0 (then renormalised) for B.
AOPC = mean drop over k = 1..5. S's evidence is the same zero-baseline IG (256 nodes), not exact there.

Shapley (`--shapley`, B and P1): on queries of at most 10 tokens, the exact Shapley value of the game that switches
tokens off with every denominator held, against the IG evidence: both sum to the same total; above degree 2 they
may differ, and how much is reported.
"""
import argparse
import itertools
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import data
import synth
from evidence import _score_fn, evidence
from models import BHR, Frozen, build

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+", help="run tags in results/")
p.add_argument("--n", type=int, default=1000)
p.add_argument("--shapley", type=int, default=0, help="queries for the Shapley comparison")
args = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
_sets = {}


def dataset(name):
    """(V, C, sample ids, sample ground-truth mask or None): the same fixed sample of test queries for every run."""
    if name not in _sets:
        d = synth.get(name)
        pick = np.random.default_rng(0).choice(len(d["x_test"]), min(args.n, len(d["x_test"])), replace=False)
        gt = torch.tensor(d["gt_test"][pick]) if "gt_test" in d else None
        _sets[name] = (len(d["words"]), len(d["classes"]), torch.tensor(d["x_test"][pick]), gt)
    return _sets[name]


def probs_fn(model):
    bhr = BHR(model.share).to(dev).double() if model.family == "B" else None
    return lambda ids: (bhr(model(ids)) if bhr is not None else F.log_softmax(model(ids), -1)).exp()


def remove(ids, order, k):
    out = ids.clone()
    out.scatter_(1, order[:, :k], 0)
    return out


def shapley(model, ids, target):
    """Exact Shapley values per token for one query, denominators held at the full query."""
    x0, f = _score_fn(model, ids)
    fz = Frozen()
    with torch.no_grad():
        f(x0, fz)
    pos = (ids[0] != 0).nonzero()[:, 0].tolist()
    n = len(pos)
    masks = torch.tensor(list(itertools.product((0, 1), repeat=n)), dtype=x0.dtype, device=x0.device)
    vals = []
    for mk in masks:
        keep = torch.zeros(ids.shape[1], dtype=x0.dtype, device=x0.device)
        keep[pos] = mk
        x = x0 * (keep if x0.dim() == 2 else keep[:, None])
        with torch.no_grad():
            vals.append(f(x, fz.start_replay())[0, target].item())
    v = dict(zip((tuple(m.int().tolist()) for m in masks), vals))
    phi = np.zeros(n)
    for m, val in v.items():
        s = sum(m)
        for i in range(n):
            if m[i] == 0:
                w = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                phi[i] += w * (v[m[:i] + (1,) + m[i + 1:]] - val)
    return pos, phi


for tag in args.runs:
    cfg = json.load(open(f"results/{tag}.json"))
    V, C, x_all, gt = dataset(cfg.get("data", "banking77"))
    model = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=x_all.shape[1]).to(dev)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    model.double().eval()
    prob = probs_fn(model)
    ids = x_all.to(dev)
    with torch.no_grad():
        p0 = prob(ids)
    target = p0.argmax(1)
    p0t = p0.gather(1, target[:, None])[:, 0]
    evs, gaps = [], []
    for i in range(0, len(ids), 100):
        ev, full, zero = evidence(model, ids[i:i + 100], target[i:i + 100])
        evs.append(ev)
        gaps.append(((ev.sum(1) - (full - zero)).abs() / (full - zero).abs().clamp(min=1e-300)))
    ev, gap = torch.cat(evs), torch.cat(gaps)
    present = ids != 0
    n_tok = present.sum(1)
    big = torch.finfo(ev.dtype).max
    orders = {
        "evidence": torch.where(present, ev, -big).argsort(1, descending=True),
        "random": torch.where(present, torch.rand(ev.shape, device=dev, dtype=ev.dtype), -big).argsort(1, descending=True),
        "lowest": torch.where(present, -ev, -big).argsort(1, descending=True),
    }
    res = {"tag": tag, "completeness_gap_median": gap.median().item(), "completeness_gap_max": gap.max().item()}
    for name, order in orders.items():
        drops, flips = [], []
        for k in range(1, 6):
            ok = n_tok > k
            with torch.no_grad():
                pk = prob(remove(ids, order, k))
            drops.append((p0t - pk.gather(1, target[:, None])[:, 0])[ok].mean().item())
            flips.append((pk.argmax(1) != target)[ok].float().mean().item())
        res[name] = dict(aopc=float(np.mean(drops)), drop=drops, flip=flips)
    line = f"{tag:22s} gap median {res['completeness_gap_median']:.1e} | AOPC evid {res['evidence']['aopc']:.3f} " \
           f"rand {res['random']['aopc']:.3f} low {res['lowest']['aopc']:.3f} | flip@1 evid {res['evidence']['flip'][0]:.2f} " \
           f"rand {res['random']['flip'][0]:.2f}"
    if gt is not None:
        # ORDER / RECALL / e-SNLI: do the k deciding (highlighted) tokens get the top k evidence ranks, k = their
        # number in the query (4 on ORDER and RECALL)? A random ranking gets k / n on average.
        g = gt.to(dev) & present
        k = g.sum(1)
        rank = torch.empty_like(orders["evidence"]).scatter_(1, orders["evidence"],
                                                             torch.arange(g.shape[1], device=dev).expand_as(g))
        topk = (rank < k[:, None]) & present
        has = k > 0
        prec = ((topk & g).sum(1).float() / k.clamp(min=1).float())[has]
        chance = (k.float() / n_tok.float())[has]
        share = (ev * g).sum(1) / ev.abs().sum(1)
        res["ground_truth"] = dict(precision_at_k=prec.mean().item(), chance=chance.mean().item(),
                                   evidence_share_on_deciding=share.mean().item())
        line += f" | deciding tokens in top k: {prec.mean().item():.3f} (chance {chance.mean().item():.3f})"
    if args.shapley and model.family != "S":
        short = ((n_tok <= 10) & (n_tok >= 2)).nonzero()[:, 0][: args.shapley].tolist()
        corr, rel = [], []
        for i in short:
            pos, phi = shapley(model, ids[i:i + 1], target[i].item())
            e = ev[i, pos].cpu().numpy()
            corr.append(np.corrcoef(e, phi)[0, 1])
            rel.append(np.abs(e - phi).sum() / np.abs(phi).sum())
        res["shapley"] = dict(n=len(short), corr_median=float(np.median(corr)), l1_rel_median=float(np.median(rel)))
        line += f" | Shapley corr {res['shapley']['corr_median']:.3f} L1 rel {res['shapley']['l1_rel_median']:.3f}"
    print(line, flush=True)
    with open(f"results/{tag}_faith.json", "w") as f:
        json.dump(res, f, indent=1)
