"""Fairer comparisons of the word rankings (referee round 2): the same 1,000 test pairs as faithfulness.py, every
method scored the same way, per-pair results kept for confidence intervals.

Methods, each a score per present token (higher = more important for the predicted class g):
  evidence   the model's own: exact quadrature (Born, Poly-sim-born); for the softmax model integrated gradients of
             its logit over the embedded vectors (256 nodes), as in the paper's tables
  gated      softmax model only: integrated gradients of its probability P_g over one scalar gate per token
             (x_t -> a_t x_t), the same path and explained quantity as BEW; Gauss-Legendre, 512 nodes
  attention  the weight each word receives (attention_agreement.py)
  attgrad    attention x gradient: sum over blocks, mean over heads, sum over present query words of
             relu(A * dP_g/dA), received by each key (a transformer-specific attribution)
  loo        leave-one-out occlusion: the drop of P_g when the token alone is deleted; its flip@1 is the ceiling
             of what any ranking can reach on flip@1 (any single deletion that flips the answer)
  random     a random order
Scores: flip@1 (deleting the top token changes the answer; pairs with > 1 token), flip@5 lowest (deleting the five
lowest; pairs with > 5 tokens), agreement (share of the top-k that annotators highlighted, k = their number; all
pairs and correctly answered pairs), and agreement on content words (referee round 5, 2026-11-12: the highlight mask
marks punctuation attached to a starred word, so '.' and the separator are taken out of both ranking and mask). Deletion is token id 0 in place (amplitude 0 for the Born model).

python fairness.py   ->  results/fairness.npz (per pair), results/fairness.json (summary with 95 % bootstrap intervals)
"""
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import synth
from evidence import _score_fn, evidence, nodes01
from models import BHR, Frozen, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)     # faithfulness.py's sample
N = int(os.environ.get("FAIR_N", 1000))                                              # a smaller N for a quick check
pick = pick[:N]
IDS = torch.tensor(d["x_test"][pick]).to(dev)
GT = torch.tensor(d["gt_test"][pick]).to(dev)
Y = torch.tensor(d["y_test"][pick]).to(dev)
PRESENT = IDS != 0
NTOK = PRESENT.sum(1)
NONWORD = torch.tensor([all(not ch.isalnum() for ch in w) or w == "<sep>" for w in d["words"]], device=dev)
CONTENT = PRESENT & ~NONWORD[IDS]
V, C = len(d["words"]), len(d["classes"])
B = min(100, N)


def probs_fn(model):
    # P_g itself for the Born models (trained on its cross-entropy; the answer is argmax P_g). Until 2026-11-10 this
    # scored BHR(P_g), Seeing Why's sharp headroom softmax: flips and agreement were the same, AOPC and the gradient
    # of attention x gradient were of BHR(P_g).
    return lambda ids: model(ids) if model.family == "B" else F.log_softmax(model(ids), -1).exp()


# ------------------------------------------------------------------------------------------- attention and gradient
def softmax_forward(model, x, ids):
    """The softmax transformer's forward written out, so the attention weights are part of the graph."""
    pad = ids == 0
    h, atts = x, []
    for b in model.blocks:
        z = b.ln1(h)
        q, k, v = (z @ b.att.in_proj_weight.T + b.att.in_proj_bias).chunk(3, -1)
        H = b.att.num_heads
        q, k, v = (t.unflatten(-1, (H, t.shape[-1] // H)) for t in (q, k, v))
        s = torch.einsum("bihe,bjhe->bhij", q, k) / math.sqrt(q.shape[-1])
        a = torch.softmax(s.masked_fill(pad[:, None, None], float("-inf")), -1)
        atts.append(a)
        h = h + b.att.out_proj(torch.einsum("bhij,bjhe->bihe", a, v).flatten(-2))
        h = h + b.mlp(b.ln2(h))
    keep = (~pad).unsqueeze(-1).to(h.dtype)
    return model.head(model.ln((h * keep).sum(1) / keep.sum(1).clamp(min=1))), atts


def attention_and_grad(model, ids, target):
    """(received attention [B, L], attention x gradient [B, L]) for P_target."""
    L = ids.shape[1]
    if model.family == "S":
        logits, atts = softmax_forward(model, model.embed(ids), ids)
        p = torch.softmax(logits, -1)
    else:
        atts = []
        orig = torch.einsum

        def ein(eq, *ops):
            if eq == "bhij,bjhe->bihe":
                atts.append(ops[0])
            return orig(eq, *ops)
        torch.einsum = ein
        try:
            p = probs_fn(model)(ids)
        finally:
            torch.einsum = orig
    for a in atts:
        a.retain_grad()
    p.gather(1, target[:, None]).sum().backward()
    q = (ids != 0)[:, None, :, None].to(torch.float64)
    recv, ag = 0, 0
    for a in atts:
        aa = a[:, :, :L, :L]
        g = a.grad[:, :, :L, :L]
        prod = (aa * g).real if aa.is_complex() else aa * g
        ar = aa.real if aa.is_complex() else aa
        recv = recv + ((ar.double() * q).sum(2) / q.sum(2).clamp(min=1)).mean(1)
        ag = ag + (prod.double().clamp(min=0) * q).sum(2).mean(1)
    return (recv / len(atts)).detach(), ag.detach()


# ---------------------------------------------------------------------------------------- gated IG (softmax model)
def gated_ig(model, ids, target, n_nodes=512):
    """IG of P_target over a scalar gate per token, a = 1 at the input and 0 at the baseline. (scores, P(1) - P(0))"""
    x = model.embed(ids).detach()
    f = lambda a: torch.softmax(model.from_embedded(x * a[..., None], ids), -1).gather(1, target[:, None])[:, 0]
    one = PRESENT.new_ones(ids.shape, dtype=x.dtype)
    with torch.no_grad():
        full, zero = f(one), f(torch.zeros_like(one))
    total = torch.zeros_like(one)
    for t, w in zip(*nodes01(n_nodes)):
        a = (float(t) * one).requires_grad_(True)
        (g,) = torch.autograd.grad(f(a).sum(), a)
        total += float(w) * g
    return (one * total).detach(), (full - zero).detach()


# ----------------------------------------------------------------------------------------------------- occlusion
def loo(prob, ids, target, p0t):
    """Drop of P_target when each present token alone is deleted [B, L] (-inf where absent), and any-flip [B]."""
    Bn, L = ids.shape
    drop = torch.full((Bn, L), float("-inf"), dtype=torch.float64, device=dev)
    flips = torch.zeros(Bn, dtype=torch.bool, device=dev)
    for j in range(L):
        has = ids[:, j] != 0
        if not has.any():
            continue
        x = ids[has].clone()
        x[:, j] = 0
        with torch.no_grad():
            pk = prob(x)
        drop[has, j] = (p0t[has] - pk.gather(1, target[has][:, None])[:, 0]).double()
        flips[has] |= pk.argmax(1) != target[has]
    return drop, flips


# ------------------------------------------------------------------------------------------------------- scoring
def score(prob, ids, target, s):
    """Per pair: flip@1, flip@5 of the five lowest, agreement (NaN where undefined)."""
    big = torch.finfo(torch.float64).max
    s = s.double()
    order = torch.where(PRESENT, s, -big).argsort(1, descending=True)
    low = torch.where(PRESENT, -s, -big).argsort(1, descending=True)
    out = {}
    with torch.no_grad():
        x1 = ids.clone().scatter_(1, order[:, :1], 0)
        f1 = (prob(x1).argmax(1) != target).double()
        out["flip1"] = torch.where(NTOK > 1, f1, torch.nan)
        x5 = ids.clone().scatter_(1, low[:, :5], 0)
        f5 = (prob(x5).argmax(1) != target).double()
        out["flip5_low"] = torch.where(NTOK > 5, f5, torch.nan)
    g = GT & PRESENT
    k = g.sum(1)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(ids.shape[1], device=dev).expand_as(order))
    topk = (rank < k[:, None]) & PRESENT
    agree = (topk & g).sum(1).double() / k.clamp(min=1).double()
    out["agree"] = torch.where(k > 0, agree, torch.nan)
    gc = GT & CONTENT                                       # content words only, in both ranking and mask
    kc = gc.sum(1)
    oc = torch.where(CONTENT, s, -big).argsort(1, descending=True)
    rc = torch.empty_like(oc).scatter_(1, oc, torch.arange(ids.shape[1], device=dev).expand_as(oc))
    out["agree_content"] = torch.where(kc > 0, ((rc < kc[:, None]) & gc).sum(1).double() / kc.clamp(min=1).double(),
                                       torch.nan)
    return out


per_pair, meta = {}, {}
# FAIR_MODELS="born:Born-ap" / FAIR_SEEDS="0,1" / FAIR_OUT=fairness_sq: another model set, written to its own files
MODELS = [tuple(m.split(":")) for m in os.environ["FAIR_MODELS"].split(",")] if "FAIR_MODELS" in os.environ else \
    [("born", "Born-ap"), ("softmax", "Softmax"), ("poly", "Poly-sim-born")]
SEEDS = [int(x) for x in os.environ.get("FAIR_SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("FAIR_OUT", "fairness")
for fam, model_name in MODELS:
    for seed in SEEDS:
        tag = f"esnli_{model_name}_b2_s{seed}_e20"
        cfg = json.load(open(f"results/{tag}.json"))
        model = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
        model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
        model.double().eval()
        prob = probs_fn(model)
        rows = {m: {"flip1": [], "flip5_low": [], "agree": [], "agree_content": []} for m in
                ("evidence", "gated", "attention", "attgrad", "loo", "random")}
        ceiling, correct, gated_gap = [], [], []
        for i in range(0, len(IDS), B):
            sl = slice(i, i + B)
            ids = IDS[sl]
            with torch.no_grad():
                p0 = prob(ids)
            t = p0.argmax(1)
            p0t = p0.gather(1, t[:, None])[:, 0]
            correct.append((t == Y[sl]).cpu())
            scores = {}
            ev, full, zero = evidence(model, ids, t)
            scores["evidence"] = ev
            if fam == "softmax":
                gs, expl = gated_ig(model, ids, t)
                scores["gated"] = gs
                gated_gap.append(((gs.sum(1) - expl).abs() / expl.abs().clamp(min=1e-300)).cpu())
            recv, ag = attention_and_grad(model, ids, t)
            scores["attention"], scores["attgrad"] = recv, ag
            drop, fl = loo(prob, ids, t, p0t)
            scores["loo"] = drop.clamp(min=-1e300)
            scores["random"] = torch.rand(ids.shape, device=dev, dtype=torch.float64)
            ceiling.append(fl.cpu())
            for m, s in scores.items():
                pres, nt, gt, ct = PRESENT, NTOK, GT, CONTENT
                PRESENT, NTOK, GT, CONTENT = pres[sl], nt[sl], gt[sl], ct[sl]  # score() reads the slice
                r = score(prob, ids, t, s)
                PRESENT, NTOK, GT, CONTENT = pres, nt, gt, ct
                for key, v in r.items():
                    rows[m][key].append(v.cpu())
        for m in rows:
            for key in rows[m]:
                per_pair[f"{fam}_s{seed}_{m}_{key}"] = torch.cat(rows[m][key]).numpy() if rows[m][key] else np.array([])
        per_pair[f"{fam}_s{seed}_ceiling"] = torch.cat(ceiling).double().numpy()
        per_pair[f"{fam}_s{seed}_correct"] = torch.cat(correct).numpy()
        if gated_gap:
            gg = torch.cat(gated_gap).numpy()
            meta[f"{fam}_s{seed}_gated_gap"] = dict(median=float(np.median(gg)), max=float(np.max(gg)))
        print(tag, "done", flush=True)

np.savez(f"results/{OUT}.npz", **per_pair)

# ---------------------------------------------------------------------------------------- summary with intervals
rng = np.random.default_rng(1)
boot = rng.integers(0, N, size=(2000, N))


def pooled(fam, m, key, only_correct=False):
    """Seed-pooled per-pair mean (mean over seeds of each pair, NaN-aware) and its 95 % bootstrap interval."""
    a = np.stack([per_pair[f"{fam}_s{s}_{m}_{key}"] for s in SEEDS])            # [seeds, 1000]
    if only_correct:
        a = np.where(np.stack([per_pair[f"{fam}_s{s}_correct"] for s in SEEDS]), a, np.nan)
    pp = np.nanmean(a, 0)
    ok = ~np.isnan(pp)
    means = [np.nanmean(pp[b][ok[b]]) for b in boot]
    return float(np.nanmean(pp[ok])), [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], pp


summary = {"meta": meta}
for name, mask in (("chance", PRESENT), ("chance_content", CONTENT)):
    kk, nn = (GT & mask).sum(1).double(), mask.sum(1).double()
    summary[name] = float((kk / nn)[kk > 0].mean())
FAMS = [f for f, _ in MODELS]
for fam in FAMS:
    summary[fam] = {}
    for m in ("evidence", "gated", "attention", "attgrad", "loo", "random"):
        if per_pair[f"{fam}_s{SEEDS[0]}_{m}_flip1"].size == 0:
            continue
        summary[fam][m] = {}
        for key in ("flip1", "flip5_low", "agree", "agree_content"):
            v, ci, _ = pooled(fam, m, key)
            summary[fam][m][key] = [v, ci]
        v, ci, _ = pooled(fam, m, "agree", only_correct=True)
        summary[fam][m]["agree_correct"] = [v, ci]
    c = np.stack([per_pair[f"{fam}_s{s}_ceiling"] for s in SEEDS]).mean(0)
    summary[fam]["ceiling_flip1"] = [float(c.mean()), [float(np.percentile([c[b].mean() for b in boot], q)) for q in (2.5, 97.5)]]

# paired differences, Born evidence against the softmax model's methods (same pairs; seed-pooled)
summary["paired_flip1"] = {}
if "born" in FAMS and "softmax" in FAMS:
    _, _, b_ev = pooled("born", "evidence", "flip1")
for m in (("evidence", "gated", "attgrad", "loo") if "born" in FAMS and "softmax" in FAMS else ()):
    _, _, s_m = pooled("softmax", m, "flip1")
    diff = b_ev - s_m
    ok = ~np.isnan(diff)
    ds = [np.nanmean(diff[b][ok[b]]) for b in boot]
    summary["paired_flip1"][f"born evidence - softmax {m}"] = [float(np.nanmean(diff[ok])),
                                                                [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]]
json.dump(summary, open(f"results/{OUT}.json", "w"), indent=1)
for fam in FAMS:
    print(fam, "ceiling flip@1 %.3f" % summary[fam]["ceiling_flip1"][0])
    for m, r in summary[fam].items():
        if m == "ceiling_flip1":
            continue
        print(f"  {m:9s} " + " | ".join(f"{k} {v[0]:.3f} [{v[1][0]:.3f}, {v[1][1]:.3f}]" for k, v in r.items()))
print(json.dumps(summary["paired_flip1"], indent=1), json.dumps(meta, indent=1))
