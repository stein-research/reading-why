"""The optical build under shot noise, over several draws, and the photons at its detectors (referee round 5,
2026-11-12). Born Transformer seed 0, first 512 test pairs, compile_hw.compiled_probs with `photons` source photons
per sentence (shared among all its detectors).

1. Accuracy at 1e8, 1e10 and 1e12 photons, --draws independent noise draws each (torch seeds 0..draws-1): mean,
   lowest and highest. (compile_hw.py --noise reports one draw, seed 0.)
2. At 1e10: the photons reaching each balanced pair, (r+ + r-) as compile_hw.balanced draws them, for the attention
   overlaps (calls 1-4: block 1 Re, Im, block 2 Re, Im) and the feed-forward readings (calls 1-2: blocks 1, 2),
   medians and 1st/99th percentiles; and at the camera.

python photon_draws.py [--draws 20]   -> results/photon_draws.json   (a few minutes on the CPU)
"""
import json
import sys

import numpy as np
import torch

import compile_hw as C

DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 20
TAG = "esnli_Born-ap_b2_s0_e20"
model, d = C.load(TAG)
ids, y = torch.tensor(d["x_test"][:512]), torch.tensor(d["y_test"][:512])
out = {"tag": TAG, "pairs": 512, "draws": DRAWS, "accuracy": {}}
with torch.no_grad():
    out["noiseless"] = float((C.compiled_probs(model, ids)[0].argmax(1) == y).double().mean())
    for lg in (8, 10, 12):
        acc = []
        for s in range(DRAWS):
            torch.manual_seed(s)
            acc.append(float((C.compiled_probs(model, ids, photons=10.0 ** lg)[0].argmax(1) == y).double().mean()))
        out["accuracy"][f"1e{lg}"] = dict(mean=float(np.mean(acc)), low=min(acc), high=max(acc), all=acc)
        print(f"1e{lg}:", out["accuracy"][f"1e{lg}"], flush=True)

rec = {"attention": [], "feed_forward": []}
orig = C.balanced


def spy(x, yy, f, photons, dims):
    def inten(z):
        z2 = z.real ** 2 + z.imag ** 2 if z.is_complex() else z ** 2
        return z2.sum(dims) if dims else z2
    share = f * f * (inten(x + yy) + inten(x - yy)) / 2          # share of the source light reaching the pair
    rec["attention" if dims == -1 else "feed_forward"].append(share.detach().reshape(share.shape[0], -1))
    return orig(x, yy, f, photons, dims)


C.balanced = spy
torch.manual_seed(0)
with torch.no_grad():
    _, power = C.compiled_probs(model, ids, photons=1e10)
C.balanced = orig
out["photons_1e10"] = {}
for k, v in rec.items():
    for i, t in enumerate(v):
        t = t[t > 0].numpy() * 1e10
        q = np.quantile(t, [0.01, 0.5, 0.99])
        out["photons_1e10"][f"{k}_call{i + 1}"] = dict(p1=float(q[0]), median=float(q[1]), p99=float(q[2]), n=int(t.size))
out["photons_1e10"]["camera"] = float(10 ** float(power) * 1e10)
print(json.dumps(out["photons_1e10"], indent=1))
json.dump(out, open("results/photon_draws.json", "w"), indent=1)
