"""The softmax model's gated evidence at its baseline (a colleague's review, 2026-11-06): with every word gate at zero
the embedded vectors are all zero and the model gives one fixed answer, so its gated integrated gradients sum to P_g
above that answer, not above the no-information share b_g as BEW does. Reports that answer per seed and the gated
flip@1 (results/gated4096.npz, 4,096 nodes) on the pairs the model answers with it and on the others, the same 1,000
test pairs as the tables. Writes results/gated_baseline.json.

python gated_baseline.py      (after gated_converged.py; seconds)
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
from models import build

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
IDS = torch.tensor(d["x_test"][pick]).to(dev)
V, C = len(d["words"]), len(d["classes"])
flip = np.load("results/gated4096.npz")

out = {"classes": d["classes"], "seeds": {}}
for seed in range(3):
    tag = f"esnli_Softmax_b2_s{seed}_e20"
    cfg = json.load(open(f"results/{tag}.json"))
    m = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    m.double().eval()
    with torch.no_grad():
        p0 = F.softmax(m.from_embedded(0 * m.embed(IDS), IDS), -1)        # every gate at zero
        t = m(IDS).argmax(1).cpu().numpy()
    base = int(p0.mean(0).argmax())
    f1 = flip[f"s{seed}_flip1"]
    s = {"p_zero": p0.mean(0).tolist(), "p_zero_spread": float(p0.std(0).max()), "baseline_answer": d["classes"][base],
         "share_answered_so": float(np.mean(t == base)),
         "gated_flip1_answered_so": float(np.nanmean(f1[t == base])),
         "gated_flip1_others": float(np.nanmean(f1[t != base]))}
    out["seeds"][seed] = s
    print(tag, json.dumps(s), flush=True)
json.dump(out, open("results/gated_baseline.json", "w"), indent=1)
