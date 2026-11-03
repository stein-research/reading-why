"""Shot-noise intervals for the device runs: the measured counts of each Hadamard test (from the ledger) are
resampled binomially and the answer recomputed through compile_hw with those overlaps (parametric bootstrap).
The circuits are the ones that ran; each draw keeps them and redraws only the 2,000 shots.

python device_bootstrap.py [--draws 1000]
"""
import json
import re
import sys

import numpy as np
import torch

from compile_hw import compiled_probs, load

tag = "esnli_Born-p2-ap_b2_s0_e20"
draws = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 1000
model, d = load(tag)
RUNS = {"pair 3763": (None, ["9333cd8540d5532d", "e5ebeafc76390413"]),
        "pair 3763, aliens deleted": (5, ["f2b655e7ea17e75a", "e39c9915ad92a7bf"])}


def readings(keys):
    """{block: {(h, i, j, part): (P0 - P1, shots)}} from the ledger's notes and counts."""
    out = {}
    for b, k in enumerate(keys):
        rec = json.load(open(f"ibm_jobs/{k}/record.json"))
        counts = json.load(open(f"ibm_jobs/{k}/counts.json"))
        for c, cnt in zip(rec["circuits"], counts):
            m = re.match(r"block (\d) head (\d) q(\d+) k(\d+) (Re|Im|calibration)", c["note"])
            blk, h, i, j, part = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
            n = sum(cnt.values())
            out.setdefault(blk, {})[(h, i, j, part)] = ((cnt.get("0", 0) - cnt.get("1", 0)) / n, n)
    return out


def overlap_from(table, rng=None):
    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        H, L = q.shape[0], q.shape[1]
        re_, im_ = torch.zeros(1, H, L, L, dtype=torch.float64), torch.zeros(1, H, L, L, dtype=torch.float64)
        for (h, i, j, part), (v, n) in table[block].items():
            if part == "calibration":
                continue
            if rng is not None:                                      # redraw the shots of this circuit
                v = 2 * rng.binomial(n, (1 + np.clip(v, -1, 1)) / 2) / n - 1
            scale = np.linalg.norm(q[h, i]) * np.linalg.norm(k[h, j])
            (re_ if part == "Re" else im_)[0, h, i, j] = v * scale
        return re_, im_
    return overlap


out = {}
for name, (delete, keys) in RUNS.items():
    ids = torch.tensor(d["x_test"][3763:3764]).clone()
    if delete is not None:
        ids[0, delete] = 0
    table = readings(keys)
    with torch.no_grad():
        p_dev = compiled_probs(model, ids, overlap=overlap_from(table))[0][0].numpy()
        rng = np.random.default_rng(0)
        ps = np.array([compiled_probs(model, ids, overlap=overlap_from(table, rng))[0][0].numpy() for _ in range(draws)])
    lo, hi = np.percentile(ps, [2.5, 97.5], axis=0)
    flip = float((ps.argmax(1) == p_dev.argmax()).mean())
    out[name] = dict(P_device=p_dev.tolist(), ci95_low=lo.tolist(), ci95_high=hi.tolist(), same_answer_share=flip,
                     draws=draws)
    print(f"{name}: P {np.round(p_dev, 3).tolist()}  95% [{np.round(lo, 3).tolist()} .. {np.round(hi, 3).tolist()}]  "
          f"answer kept in {flip:.3f} of draws", flush=True)
json.dump(out, open("results/device_bootstrap_3763.json", "w"), indent=1)
