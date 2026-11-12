"""The bag of embeddings' own evidence on faithfulness.py's 1,000 e-SNLI test pairs (referee round 7, 2026-11-12):
flip@1, share of the leave-one-out ceiling, flip@5 lowest and agreement for gradient x input, integrated gradients of
its probability (1,024 nodes), leave-one-out and random, for bag_baseline.py's inter model, seed 0.
python bag_evidence.py   (after bag_baseline.py inter 0 20; GPU, a few minutes) -> results/bag_evidence.json (printed lines)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import synth
d = synth.get("esnli"); V = len(d["words"]); dev = "cuda"
class DAN(nn.Module):
    def __init__(s, dim=64, hid=300):
        super().__init__(); s.emb = nn.Embedding(V, dim); s.mlp = nn.Sequential(nn.Linear(4*dim, hid), nn.ReLU(), nn.Linear(hid, 3))
    def masks(s, ids):
        pres = ids != 0; sep = (ids == 2).int().argmax(1); pos = torch.arange(ids.shape[1], device=ids.device)[None]; hassep = (ids == 2).any(1)
        mp = (pres & (pos < sep[:, None]) & hassep[:, None]).double(); mh = (pres & ((pos > sep[:, None]) | ~hassep[:, None]) & (ids != 2)).double()
        return mp, mh
    def from_e(s, e, ids):
        mp, mh = s.masks(ids)
        u = (e * mp[..., None]).sum(1) / mp.sum(1, keepdim=True).clamp(min=1); v = (e * mh[..., None]).sum(1) / mh.sum(1, keepdim=True).clamp(min=1)
        return s.mlp(torch.cat((u, v, u * v, (u - v).abs()), 1))
    def forward(s, ids): return s.from_e(s.emb(ids), ids)
m = DAN().to(dev); m.load_state_dict(torch.load("results/bag_inter_s0.pt")); m.double().eval()
with torch.no_grad():
    xa, ya = torch.tensor(d["x_test"], device=dev), torch.tensor(d["y_test"], device=dev)
    print("test acc", (m(xa).argmax(1) == ya).double().mean().item())
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
ids = torch.tensor(d["x_test"][pick], device=dev); gt = torch.tensor(d["gt_test"][pick], device=dev)
prob = lambda i: F.softmax(m(i), -1)
with torch.no_grad(): p0 = prob(ids)
target = p0.argmax(1); present = ids != 0; n_tok = present.sum(1); L = ids.shape[1]
# leave-one-out
loo = torch.full(ids.shape, -1e9, dtype=torch.double, device=dev); flipany = torch.zeros(len(ids), dtype=torch.bool, device=dev)
with torch.no_grad():
    for t in range(L):
        x = ids.clone(); x[:, t] = 0; p = prob(x); ok = present[:, t]
        loo[ok, t] = (p0.gather(1, target[:, None])[:, 0] - p.gather(1, target[:, None])[:, 0])[ok]
        flipany |= ok & (p.argmax(1) != target)
# IG of probability (and logit) over embeddings, zero baseline, 1024 midpoint nodes; gradient x input
e0 = m.emb(ids).detach()
def ig(fn, nodes=1024):
    tot = torch.zeros_like(e0)
    for a in (torch.arange(nodes, dtype=torch.double) + 0.5) / nodes:
        e = (a * e0).requires_grad_(True); (g,) = torch.autograd.grad(fn(e).gather(1, target[:, None]).sum(), e); tot += g / nodes
    return (e0 * tot).sum(-1)
fP = lambda e: F.softmax(m.from_e(e, ids), -1); fL = lambda e: m.from_e(e, ids)
igP, igL = ig(fP), ig(fL)
with torch.no_grad(): gapP = (igP.sum(1) - (fP(e0) - fP(0 * e0)).gather(1, target[:, None])[:, 0]).abs()
print("IG(P) completeness gap median/max", gapP.median().item(), gapP.max().item())
e = e0.clone().requires_grad_(True); (g,) = torch.autograd.grad(fP(e).gather(1, target[:, None]).sum(), e); gxi = (e0 * g).sum(-1)
big = 1e300
def score(name, ev):
    order = torch.where(present, ev, torch.full_like(ev, -big)).argsort(1, descending=True)
    low = torch.where(present, -ev, torch.full_like(ev, -big)).argsort(1, descending=True)
    with torch.no_grad():
        x = ids.clone(); x.scatter_(1, order[:, :1], 0); f1 = (prob(x).argmax(1) != target)[n_tok > 1].double().mean().item()
        x = ids.clone(); x.scatter_(1, low[:, :5], 0); f5 = (prob(x).argmax(1) != target)[n_tok > 5].double().mean().item()
    g = gt & present; k = g.sum(1); rank = torch.empty_like(order).scatter_(1, order, torch.arange(L, device=dev).expand_as(order))
    topk = (rank < k[:, None]) & present; has = k > 0
    agree = ((topk & g).sum(1).double() / k.clamp(min=1).double())[has].mean().item()
    print(f"  {name:28s} flip@1 {f1:.3f}  of ceiling {f1/flipany.double().mean().item():.2f}  flip@5 lowest {f5:.3f}  agreement {agree:.3f}")
print("ceiling (some single deletion flips):", flipany.double().mean().item())
torch.manual_seed(0)
score("IG of P, embeddings", igP); score("IG of logit, embeddings", igL); score("gradient x input (P)", gxi); score("leave-one-out", loo)
score("random", torch.rand(ids.shape, dtype=torch.double, device=dev))

_j.dump(dict(script="bag_evidence.py", lines=_LINES), open("results/bag_evidence.json", "w"), indent=1)
