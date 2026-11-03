"""Gradient x input for every model (referee: the simplest attribution baseline), scored as in fairness.py on the
same 1,000 test pairs: flip@1, flip@5 of the five lowest, agreement; pooled over three seeds per pair, 95 % bootstrap
intervals. One scalar per word, of P_g: the Born Transformer's amplitude a_t (its input), a gate a_t = 1 on the
embedded vector for the others, so score_t = a_t dP_g/da_t at the input (integrated gradients with one node at the
end of the path).

python gradinput.py  -> results/gradinput.json (and merged into results/fairness.json as method 'gradinput')
"""
import json

import numpy as np
import torch
import torch.nn.functional as F

import synth
from models import build

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
IDS = torch.tensor(d["x_test"][pick]).to(dev)
GT = torch.tensor(d["gt_test"][pick]).to(dev)
Y = torch.tensor(d["y_test"][pick]).to(dev)
V, C = len(d["words"]), len(d["classes"])


def prob_fn(model):
    if model.family == "B":
        return lambda ids: model(ids)
    return lambda ids: F.softmax(model(ids), -1)


def grad_input(model, ids, target):
    if model.family == "B":
        a = model.amplitudes(ids).detach().requires_grad_(True)
        num, z = model.numerators(ids, a)
        p = num / (z[:, None] + 1e-300) + model.share
    else:
        x = model.embed(ids).detach()
        a = torch.ones(ids.shape, dtype=x.dtype, device=dev, requires_grad=True)
        p = F.softmax(model.from_embedded(x * a[..., None], ids), -1)
    (g,) = torch.autograd.grad(p.gather(1, target[:, None]).sum(), a)
    return (a * g).detach()


def score(prob, ids, target, s, present, ntok, gt):
    big = torch.finfo(torch.float64).max
    s = s.double()
    order = torch.where(present, s, -big).argsort(1, descending=True)
    low = torch.where(present, -s, -big).argsort(1, descending=True)
    with torch.no_grad():
        f1 = (prob(ids.clone().scatter_(1, order[:, :1], 0)).argmax(1) != target).double()
        f5 = (prob(ids.clone().scatter_(1, low[:, :5], 0)).argmax(1) != target).double()
    g = gt & present
    k = g.sum(1)
    rank = torch.empty_like(order).scatter_(1, order, torch.arange(ids.shape[1], device=dev).expand_as(order))
    agree = (((rank < k[:, None]) & present & g).sum(1).double() / k.clamp(min=1).double())
    return dict(flip1=torch.where(ntok > 1, f1, torch.nan), flip5_low=torch.where(ntok > 5, f5, torch.nan),
                agree=torch.where(k > 0, agree, torch.nan))


per = {}
for fam, name in (("born", "Born-p2-ap"), ("softmax", "Softmax"), ("poly", "Poly-sim-born")):
    for seed in range(3):
        tag = f"esnli_{name}_b2_s{seed}_e20"
        cfg = json.load(open(f"results/{tag}.json"))
        model = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
        model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
        model.double().eval()
        prob = prob_fn(model)
        rows = {k: [] for k in ("flip1", "flip5_low", "agree")}
        correct = []
        for i in range(0, len(IDS), 100):
            ids = IDS[i:i + 100]
            with torch.no_grad():
                t = prob(ids).argmax(1)
            correct.append((t == Y[i:i + 100]).cpu())
            s = grad_input(model, ids, t)
            r = score(prob, ids, t, s, ids != 0, (ids != 0).sum(1), GT[i:i + 100])
            for k, v in r.items():
                rows[k].append(v.cpu())
        for k in rows:
            per[f"{fam}_s{seed}_{k}"] = torch.cat(rows[k]).numpy()
        per[f"{fam}_s{seed}_correct"] = torch.cat(correct).numpy()
        print(tag, "done", flush=True)

rng = np.random.default_rng(1)
boot = rng.integers(0, 1000, size=(2000, 1000))
fair = json.load(open("results/fairness.json"))
out = {}
for fam in ("born", "softmax", "poly"):
    out[fam] = {}
    for key in ("flip1", "flip5_low", "agree", "agree_correct"):
        k2 = "agree" if key == "agree_correct" else key
        a = np.stack([per[f"{fam}_s{s}_{k2}"] for s in range(3)])
        if key == "agree_correct":
            a = np.where(np.stack([per[f"{fam}_s{s}_correct"] for s in range(3)]), a, np.nan)
        pp = np.nanmean(a, 0)
        ok = ~np.isnan(pp)
        ms = [np.nanmean(pp[b][ok[b]]) for b in boot]
        out[fam][key] = [float(np.nanmean(pp[ok])), [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]]
    ceil = fair[fam]["ceiling_flip1"][0]
    out[fam]["of_ceiling"] = out[fam]["flip1"][0] / ceil
    fair[fam]["gradinput"] = {k: v for k, v in out[fam].items() if k != "of_ceiling"}
    print(fam, {k: (round(v[0], 3) if isinstance(v, list) else round(v, 3)) for k, v in out[fam].items()})
json.dump(out, open("results/gradinput.json", "w"), indent=1)
json.dump(fair, open("results/fairness.json", "w"), indent=1)
