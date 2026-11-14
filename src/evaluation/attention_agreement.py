"""Raw attention against the annotators (e-SNLI): do the words a model attends to most match the words people
highlighted? The same fixed 1,000 test pairs and the same score as the evidence's agreement in faithfulness.py: the
share of the top-k words that annotators highlighted, k = the number highlighted in the pair (chance k / n).

A word's attention score is the weight it receives, averaged over the present query words and the heads: per block
("block 1", "block 2"), averaged over blocks ("mean"), and attention rollout over blocks (Abnar & Zuidema 2020:
0.5 A + 0.5 I per block, multiplied, then received). Softmax reads nn.MultiheadAttention's weights; the Born and
Poly-sim-born models the row-normalised weights w of every block on a plain forward (the Born model's reference
slot, one key past the words, is dropped). The same scores also order a deletion, as faithfulness.py does for the
evidence: remove the top 1-5 words by attention and record the drop of the predicted class's probability (AOPC) and
how often the answer changes (flip@k). Writes results/attention_agreement.json.

python attention_agreement.py <run tags...>
"""
import json
import pathlib
import sys

import numpy as np
import torch

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import synth
import torch.nn.functional as F

from models import BHR, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), min(1000, len(d["x_test"])), replace=False)
ids = torch.tensor(d["x_test"][pick]).to(dev)
gt = torch.tensor(d["gt_test"][pick]).to(dev)
present = ids != 0
g = gt & present
k = g.sum(1)
n_tok = present.sum(1)
has = k > 0


def agreement(score):
    big = torch.finfo(score.dtype).max
    order = torch.where(present, score, -big).argsort(1, descending=True)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(ids.shape[1], device=dev).expand_as(order))
    topk = (rank < k[:, None]) & present
    return ((topk & g).sum(1).float() / k.clamp(min=1).float())[has].mean().item()


def weights(model):
    """Per block, attention [B, H, L, L] over the words."""
    rec = []
    if model.family == "S":
        orig = torch.nn.MultiheadAttention.forward

        def fwd(self, *a, **kw):
            kw.update(need_weights=True, average_attn_weights=False)
            out = orig(self, *a, **kw)
            rec.append(out[1])
            return out
        torch.nn.MultiheadAttention.forward = fwd
        try:
            with torch.no_grad():
                model(ids)
        finally:
            torch.nn.MultiheadAttention.forward = orig
    else:
        orig = torch.einsum

        def ein(eq, *ops):
            if eq == "bhij,bjhe->bihe":
                rec.append(ops[0].real if ops[0].is_complex() else ops[0])
            return orig(eq, *ops)
        torch.einsum = ein
        try:
            with torch.no_grad():
                model(ids)
        finally:
            torch.einsum = orig
    L = ids.shape[1]
    return [w[:, :, :L, :L].double() for w in rec]


def probs_fn(model):
    # P_g itself for the Born models (trained on its cross-entropy; the answer is argmax P_g). Until 2026-11-10 this
    # scored BHR(P_g), Seeing Why's sharp headroom softmax: flips and agreement were the same, AOPC and the gradient
    # of attention x gradient were of BHR(P_g).
    return lambda x: model(x) if model.family == "B" else F.log_softmax(model(x), -1).exp()


def deletion(model, score):
    prob = probs_fn(model)
    big = torch.finfo(score.dtype).max
    order = torch.where(present, score, -big).argsort(1, descending=True)
    with torch.no_grad():
        p0 = prob(ids)
        t = p0.argmax(1)
        p0t = p0.gather(1, t[:, None])[:, 0]
        drops, flips = [], []
        for kk in range(1, 6):
            ok = n_tok > kk
            pk = prob(ids.clone().scatter_(1, order[:, :kk], 0))
            drops.append((p0t - pk.gather(1, t[:, None])[:, 0])[ok].mean().item())
            flips.append((pk.argmax(1) != t)[ok].float().mean().item())
    return float(np.mean(drops)), flips[0]


res = {}
chance = (k.float() / n_tok.float())[has].mean().item()
for tag in sys.argv[1:]:
    cfg = json.load(open(f"results/{tag}.json"))
    model = build(cfg["model"], len(d["words"]), len(d["classes"]), blocks=cfg["blocks"], max_len=ids.shape[1]).to(dev)
    model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    model.eval()
    ws = weights(model)
    assert len(ws) == cfg["blocks"], (tag, len(ws))
    q = present[:, None, :, None].double()                     # average over present query words
    recv = [((w * q).sum(2) / q.sum(2).clamp(min=1)).mean(1) for w in ws]     # [B, L]
    eye = torch.eye(ids.shape[1], device=dev, dtype=torch.double)
    roll = None
    for w in ws:
        a = 0.5 * w.mean(1) + 0.5 * eye
        a = a / a.sum(-1, keepdim=True)
        roll = a if roll is None else a @ roll
    out = {f"block {i + 1}": agreement(r) for i, r in enumerate(recv)}
    out["mean"] = agreement(sum(recv) / len(recv))
    out["rollout"] = agreement((roll * present[:, :, None].double()).sum(1) / present.sum(1, keepdim=True).double())
    fa = json.load(open(f"results/{tag}_faith.json"))
    out["evidence"] = fa["ground_truth"]["precision_at_k"]
    model.double()
    ws = weights(model)
    recv = [((w * q).sum(2) / q.sum(2).clamp(min=1)).mean(1) for w in ws]
    out["aopc_attention"], out["flip1_attention"] = deletion(model, sum(recv) / len(recv))
    out["aopc_evidence"], out["flip1_evidence"] = fa["evidence"]["aopc"], fa["evidence"]["flip"][0]
    out["aopc_random"], out["flip1_random"] = fa["random"]["aopc"], fa["random"]["flip"][0]
    res[tag] = out
    print(f"{tag:34s} " + " ".join(f"{a} {b:.3f}" for a, b in out.items()), flush=True)
res["chance"] = chance
print(f"chance {chance:.3f}")
json.dump(res, open("results/attention_agreement.json", "w"), indent=1)
