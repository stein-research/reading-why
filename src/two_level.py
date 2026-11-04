"""Two-level bootstrap for Table 2 (referee round 3, M3): resample the three seeds (with replacement) and the 1,000
pairs, so intervals carry the seed-to-seed spread as well; plus each method's mean over seeds and its sample SD.
Reads results/fairness.npz (and results/gated4096.npz when present). Writes results/two_level.json.
"""
import json
import os

import numpy as np

P = dict(np.load("results/fairness.npz"))
if os.path.exists("results/gated4096.npz"):
    G = np.load("results/gated4096.npz")
    for s in range(3):
        for k in ("flip1", "flip5_low", "agree"):
            P[f"softmax_s{s}_gated4096_{k}"] = G[f"s{s}_{k}"]
rng = np.random.default_rng(2)
B = 4000


def arr(fam, m, key):
    return np.stack([P[f"{fam}_s{s}_{m}_{key}"] for s in range(3)])      # [3 seeds, 1000 pairs]


def two_level(a):
    out = []
    for _ in range(B):
        ss = rng.integers(0, 3, 3)
        pp = rng.integers(0, a.shape[1], a.shape[1])
        out.append(np.nanmean(a[ss][:, pp]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


res = {"methods": {}, "differences": {}}
for fam in ("born", "softmax", "poly"):
    for m in ("evidence", "gated", "gated4096", "attention", "attgrad", "loo", "random"):
        if f"{fam}_s0_{m}_flip1" not in P or P[f"{fam}_s0_{m}_flip1"].size == 0:
            continue
        for key in ("flip1", "agree"):
            a = arr(fam, m, key)
            per_seed = np.nanmean(a, 1)
            res["methods"][f"{fam} {m} {key}"] = dict(mean=float(per_seed.mean()), seed_sd=float(per_seed.std(ddof=1)),
                                                       ci_two_level=two_level(a))
pairs = [("born", "evidence", "softmax", "gated4096" if "softmax_s0_gated4096_flip1" in P else "gated", "flip1"),
         ("born", "evidence", "softmax", "attgrad", "flip1"),
         ("born", "evidence", "born", "attention", "flip1"),
         ("born", "evidence", "softmax", "evidence", "agree"),
         ("born", "evidence", "softmax", "gated4096" if "softmax_s0_gated4096_flip1" in P else "gated", "agree"),
         ("softmax", "attention", "born", "evidence", "agree"),
         ("born", "evidence", "born", "attention", "agree"),
         ("softmax", "attgrad", "born", "evidence", "agree")]
for f1, m1, f2, m2, key in pairs:
    d = arr(f1, m1, key) - arr(f2, m2, key)              # seeds are matched by index only; models are independent
    res["differences"][f"{f1} {m1} - {f2} {m2} ({key})"] = dict(mean=float(np.nanmean(d)), ci_two_level=two_level(d))
json.dump(res, open("results/two_level.json", "w"), indent=1)
for k, v in res["differences"].items():
    print(f"{k:55s} {v['mean']:+.3f} [{v['ci_two_level'][0]:+.3f}, {v['ci_two_level'][1]:+.3f}]")
for k, v in res["methods"].items():
    print(f"{k:32s} {v['mean']:.3f} sd {v['seed_sd']:.3f} [{v['ci_two_level'][0]:.3f}, {v['ci_two_level'][1]:.3f}]")
