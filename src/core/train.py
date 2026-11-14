"""Train one model on BANKING77 (or the generated ORDER task, `--data order`) and report test accuracy.

Same recipe for every family: Adam, cosine learning rate, batch 64, gradient norm clipped at 1, token dropout
(each token removed with probability `--drop` in training, at least one kept; for B that is amplitude 0, a valid
physical input). S and P1 train on the cross-entropy of their logits, B on the cross-entropy of the Born headroom
readout (pt 1.5, phi = 0.5 / (G - 1), no trained parameter). Accuracy is the last epoch's, no model selection.
Writes results/<tag>.json (and .pt).
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import data
import synth
from compile_hw import compiled_probs
from models import BHR, build, count

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--blocks", type=int, default=1)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=2e-3)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--drop", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--data", default="banking77", help="banking77, order, recall<n>")
p.add_argument("--loss", default="born_ce", choices=["born_ce", "bhr"],
               help="B only: the plain cross-entropy of P_g (default since round 4), or BHR (ablation; rounds 1-3)")
p.add_argument("--power", type=float, default=0.0,
               help="B only: penalty per decade of light lost when compiled to passive hardware (compile_hw.py)")
p.add_argument("--tag", default="")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get(args.data)
xtr, ytr = torch.tensor(d["x_train"]), torch.tensor(d["y_train"])
xte, yte = torch.tensor(d["x_test"], device=dev), torch.tensor(d["y_test"], device=dev)
# a dev split (e-SNLI) is what a learning rate or length is chosen on; test only reports
xdv, ydv = ((torch.tensor(d["x_dev"], device=dev), torch.tensor(d["y_dev"], device=dev)) if "x_dev" in d
            else (None, None))
V, C = len(d["words"]), len(d["classes"])
model = build(args.model, V, C, blocks=args.blocks, max_len=xtr.shape[1]).to(dev)
bhr = BHR(model.share).to(dev) if model.family == "B" and getattr(model, "lin_head", None) is None else None
total, non_emb = count(model)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
steps = args.epochs * math.ceil(len(xtr) / args.batch)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))


def log_probs(ids):
    out = model(ids)
    if bhr is None:
        return F.log_softmax(out, -1)
    return bhr(out) if args.loss == "bhr" else torch.log(out.clamp(min=1e-12))


def power_lost():
    """log10 of the light a compiled pass keeps, on the first 256 test queries (B only)."""
    if model.family != "B" or getattr(model, "lin_head", None) is not None or not getattr(model, "compilable", True):
        return None
    model.eval()
    with torch.no_grad():
        return float(compiled_probs(model, xte[:256])[1])


def accuracy(x, y):
    model.eval()
    hits = 0
    with torch.no_grad():
        for i in range(0, len(x), 512):
            hits += (log_probs(x[i:i + 512]).argmax(1) == y[i:i + 512]).sum().item()
    model.train()
    return hits / len(x)


tag = args.tag or ("" if args.data == "banking77" else f"{args.data}_") + f"{args.model}_b{args.blocks}_s{args.seed}"
print(f"{tag}: params {total} (non-embedding {non_emb}), {dev}", flush=True)
t0, hist = time.time(), []
for ep in range(args.epochs):
    perm = torch.randperm(len(xtr))
    run = 0.0
    for i in range(0, len(xtr), args.batch):
        idx = perm[i:i + args.batch]
        ids = xtr[idx].clone()
        if args.drop > 0:
            cut = (torch.rand(ids.shape) < args.drop) & (ids != 0)
            keep_one = (ids != 0) & ~cut
            cut[keep_one.sum(1) == 0] = False
            ids[cut] = 0
        ids, y = ids.to(dev), ytr[idx].to(dev)
        loss = F.nll_loss(log_probs(ids), y)
        if args.power > 0:
            loss = loss - args.power * compiled_probs(model, ids)[1]          # power is log10 of a factor <= 1
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item() * len(idx)
    acc = accuracy(xte, yte)
    hist.append(dict(epoch=ep + 1, loss=run / len(xtr), test_acc=acc))
    dv = ""
    if xdv is not None:
        hist[-1]["dev_acc"] = accuracy(xdv, ydv)
        dv = f" dev {hist[-1]['dev_acc']:.4f}"
    print(f"epoch {ep + 1:3d} loss {run / len(xtr):.4f}{dv} test {acc:.4f} ({time.time() - t0:.0f} s)", flush=True)

os.makedirs("results", exist_ok=True)
res = dict(vars(args), tag=tag, power_lost_log10=power_lost(), params=total, params_non_embedding=non_emb, test_acc=hist[-1]["test_acc"],
           dev_acc=hist[-1].get("dev_acc"),
           train_acc=accuracy(xtr.to(dev), ytr.to(dev)), seconds=time.time() - t0, history=hist)
with open(f"results/{tag}.json", "w") as f:
    json.dump(res, f, indent=1)
torch.save(model.state_dict(), f"results/{tag}.pt")
print(f"{tag}: " + (f"dev {res['dev_acc']:.4f} " if xdv is not None else "") +
      f"test {res['test_acc']:.4f} train {res['train_acc']:.4f}")
