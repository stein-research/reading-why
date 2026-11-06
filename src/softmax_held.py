"""The softmax model's gated IG with its denominators held (a colleague's review, 2026-11-06): BEW holds every
denominator at the input, and with the denominators held gradient x input already ranks words much as BEW does
(path_check.py), so the like-for-like comparison holds the softmax model's denominators too: each attention row's
sum of exponentials, each LayerNorm's standard deviation and the output softmax's sum, all at their values for the
input. The model is still not a polynomial (exp, GELU), and with every gate at zero the held score is not zero
(exp(0) = 1, the biases), so its IG sums to F(1) - F(0), by Gauss-Legendre quadrature with --nodes points; the gap is
reported. The held forward at gate 1 is checked against the model. The held path overflows float64 on most
pairs (exp of exp through two blocks: the input's row sums are far below the exponentials met along the path), so
the rankings are scored on the pairs whose evidence is finite and complete (relative gap below 1e-3), and every
other ranking (gated IG with the denominators free, 4,096 nodes; the Born Transformer's and Poly-sim-born's exact
evidence) is scored on the same pairs, from results/gated4096.npz and results/fairness.npz. Same 1,000 test pairs as the tables; deletion on
the model itself, as for every ranking. Writes results/softmax_held.json (per pair results/softmax_held.npz).

python softmax_held.py [--nodes 1024]
"""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

import synth
from evidence import nodes01
from models import build

dev = "cuda" if torch.cuda.is_available() else "cpu"
NODES = int(sys.argv[sys.argv.index("--nodes") + 1]) if "--nodes" in sys.argv else 1024
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
IDS = torch.tensor(d["x_test"][pick]).to(dev)
GT = torch.tensor(d["gt_test"][pick]).to(dev)
V, C = len(d["words"]), len(d["classes"])


def layer_norm(ln, x, held, rec):
    mu = x.mean(-1, keepdim=True)
    sd = torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + ln.eps)
    if rec is not None:
        rec.append(sd.detach())
    else:
        sd = held.pop(0)
    return (x - mu) / sd * ln.weight + ln.bias


def attention(att, z, pad, held, rec):
    B, L, dm = z.shape
    h = att.num_heads
    q, k, v = F.linear(z, att.in_proj_weight, att.in_proj_bias).chunk(3, -1)
    q, k, v = (t.view(B, L, h, dm // h).transpose(1, 2) for t in (q, k, v))
    s = (q @ k.transpose(-1, -2)) / (dm // h) ** 0.5
    s = s.masked_fill(pad[:, None, None, :], float("-inf"))
    if rec is not None:
        lz = torch.logsumexp(s, -1, keepdim=True)
        rec.append(lz.detach())
    else:
        lz = held.pop(0)
    y = (torch.exp(s - lz) @ v).transpose(1, 2).reshape(B, L, dm)
    return att.out_proj(y)


def prob(m, x, ids, held=None, rec=None):
    """P of every class; rec: a list that records the denominators; held: a copy of it to replay."""
    pad = ids == 0
    hh = x
    for b in m.blocks:
        hh = hh + attention(b.att, layer_norm(b.ln1, hh, held, rec), pad, held, rec)
        hh = hh + b.mlp(layer_norm(b.ln2, hh, held, rec))
    keep = (~pad).unsqueeze(-1).to(hh.dtype)
    logit = m.head(layer_norm(m.ln, (hh * keep).sum(1) / keep.sum(1).clamp(min=1), held, rec))
    if rec is not None:
        lz = torch.logsumexp(logit, -1, keepdim=True)
        rec.append(lz.detach())
    else:
        lz = held.pop(0)
    return torch.exp(logit - lz)


def held_ig(m, ids, target, chunk=64):
    x = m.embed(ids).detach()
    rec = []
    with torch.no_grad():
        full = prob(m, x, ids, rec=rec)
    check = (full - F.softmax(m(ids), -1)).abs().max().item()
    rep = lambda r, n: [t.repeat(n, *[1] * (t.dim() - 1)) for t in r]
    with torch.no_grad():
        zero = prob(m, 0 * x, ids, held=list(rec)).gather(1, target[:, None])[:, 0]
    full = full.gather(1, target[:, None])[:, 0]
    t, w = nodes01(NODES)
    total = torch.zeros(ids.shape, dtype=x.dtype, device=dev)
    for i in range(0, NODES, chunk):
        tt = torch.tensor(t[i:i + chunk], dtype=x.dtype, device=dev)
        ww = torch.tensor(w[i:i + chunk], dtype=x.dtype, device=dev)
        n = len(tt)
        a = (tt[:, None, None] * torch.ones(ids.shape, dtype=x.dtype, device=dev)[None]).reshape(-1, ids.shape[1])
        a.requires_grad_(True)
        out = prob(m, x.repeat(n, 1, 1) * a[..., None], ids.repeat(n, 1), held=rep(rec, n))
        (g,) = torch.autograd.grad(out.gather(1, target.repeat(n)[:, None]).sum(), a)
        total += (ww[:, None, None] * g.view(n, *ids.shape)).sum(0)
    return total.detach(), (full - zero).detach(), zero.detach(), check


def score(model_prob, ids, target, s, gt):
    present = ids != 0
    ntok = present.sum(1)
    big = torch.finfo(torch.float64).max
    order = torch.where(present, s, -big).argsort(1, descending=True)
    low = torch.where(present, -s, -big).argsort(1, descending=True)
    with torch.no_grad():
        f1 = (model_prob(ids.clone().scatter_(1, order[:, :1], 0)).argmax(1) != target).double()
        f5 = (model_prob(ids.clone().scatter_(1, low[:, :5], 0)).argmax(1) != target).double()
    g = gt & present
    k = g.sum(1)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(ids.shape[1], device=dev).expand_as(order))
    agree = ((rank < k[:, None]) & present & g).sum(1).double() / k.clamp(min=1).double()
    return (torch.where(ntok > 1, f1, torch.nan), torch.where(ntok > 5, f5, torch.nan),
            torch.where(k > 0, agree, torch.nan))


per, out = {}, {"nodes": NODES, "seeds": {}}
fair, g4096 = np.load("results/fairness.npz"), np.load("results/gated4096.npz")
for seed in range(3):
    tag = f"esnli_Softmax_b2_s{seed}_e20"
    cfg = json.load(open(f"results/{tag}.json"))
    m = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    m.double().eval()
    model_prob = lambda ids: F.softmax(m(ids), -1)
    rows = {k: [] for k in ("flip1", "flip5_low", "agree", "ok")}
    checks = []
    for i in range(0, 1000, 20):
        ids = IDS[i:i + 20]
        with torch.no_grad():
            t = model_prob(ids).argmax(1)
        s, expl, zero, check = held_ig(m, ids, t)
        checks.append(check)
        ok = torch.isfinite(s).all(1) & torch.isfinite(expl) & ((s.sum(1) - expl).abs() < 1e-3 * expl.abs())
        s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        for k, v in zip(("flip1", "flip5_low", "agree"), score(model_prob, ids, t, s, GT[i:i + 20])):
            rows[k].append(v.cpu())
        rows["ok"].append(ok.cpu())
    for k in rows:
        per[f"s{seed}_{k}"] = torch.cat(rows[k]).numpy()
    ok = per[f"s{seed}_ok"]
    o = {"share_ok": float(ok.mean()), "forward_check": float(max(checks))}
    for k in ("flip1", "flip5_low", "agree"):
        o[f"held_{k}"] = float(np.nanmean(per[f"s{seed}_{k}"][ok]))
        o[f"free_{k}"] = float(np.nanmean(g4096[f"s{seed}_{k}"][ok]))
        o[f"born_{k}"] = float(np.nanmean(fair[f"born_s{seed}_evidence_{k}"][ok]))
        o[f"poly_{k}"] = float(np.nanmean(fair[f"poly_s{seed}_evidence_{k}"][ok]))
    out["seeds"][seed] = o
    print(tag, json.dumps(o), flush=True)
np.savez("results/softmax_held.npz", **per)
for k in out["seeds"][0]:
    out[k] = float(np.mean([out["seeds"][s][k] for s in range(3)]))
print(json.dumps({k: v for k, v in out.items() if k != "seeds"}, indent=1))
json.dump(out, open("results/softmax_held.json", "w"), indent=1)
