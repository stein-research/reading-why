"""Bag-of-words references: how much of BANKING77 is keywords? Logistic regression on word counts (no order,
no attention), and the train/test overlap of exact queries. Same vocabulary and tokenizer as the transformers."""
import numpy as np
import torch
import torch.nn.functional as F

import sys

import data
import synth

name = sys.argv[sys.argv.index("--data") + 1] if "--data" in sys.argv else "banking77"
d = synth.get(name)
seen = {tuple(r) for r in d["x_train"]}
print(f"test queries identical to a training query (after tokenizing): {np.mean([tuple(r) in seen for r in d['x_test']]):.4f}")
V, C = len(d["words"]), len(d["classes"])


dev = "cuda" if torch.cuda.is_available() else "cpu"


def bag(x, part):
    """Token ids for a sum-of-embeddings bag (= a linear layer on word counts, without the count matrix). part:
    "both" one bag; on e-SNLI also "split" (premise and hypothesis words told apart) and "hyp" (hypothesis only)."""
    x = torch.tensor(x)
    if part == "both":
        return x
    after = (x == 2).long().cumsum(1) > 0                                  # 2 = <sep>
    if part == "hyp":
        return torch.where(after, x, torch.zeros_like(x))
    return torch.where(after & (x != 0), x + V, x)


big = len(d["x_train"]) > 100000
parts = ("both", "split", "hyp") if name == "esnli" else ("both",)
for part in parts:
    Xtr, Xte = bag(d["x_train"], part), bag(d["x_test"], part).to(dev)
    ytr, yte = torch.tensor(d["y_train"]), torch.tensor(d["y_test"], device=dev)
    for wd in ((0.0,) if big else (0.0, 1e-4, 1e-3)):
        accs = []
        for seed in range(1 if big else 3):
            torch.manual_seed(seed)
            lin = torch.nn.EmbeddingBag(2 * V, C, mode="sum", padding_idx=0).to(dev)
            bias = torch.nn.Parameter(torch.zeros(C, device=dev))
            with torch.no_grad():
                lin.weight.uniform_(-1 / V ** 0.5, 1 / V ** 0.5)
            opt = torch.optim.Adam(list(lin.parameters()) + [bias], lr=1e-2, weight_decay=wd)
            for ep in range(5 if big else 30):
                for idx in torch.randperm(len(Xtr)).split(64 if not big else 512):
                    loss = F.cross_entropy(lin(Xtr[idx].to(dev)) + bias, ytr[idx].to(dev))
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                accs.append(((lin(Xte) + bias).argmax(1) == yte).float().mean().item())
        print(f"bag of words ({part}) + logistic regression, weight decay {wd:g}: "
              f"{100*np.mean(accs):.2f} +- {100*np.std(accs):.2f}", flush=True)
