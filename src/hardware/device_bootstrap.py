"""Shot-noise intervals for the device runs: the measured counts of each Hadamard test (from the ledger) are
resampled binomially and the answer recomputed through compile_hw with those overlaps (parametric bootstrap).
The circuits are the ones that ran; each draw keeps them and redraws only the 2,000 shots.

python device_bootstrap.py [--tag esnli_Born-ap_b2_s0_e20] [--draws 1000] [--only "pair 3765"]
    # --only: that run, merged into the saved file; the random-pair calibrations of a --cal-random run are skipped
"""
import json
import pathlib
import re
import sys

import numpy as np
import torch

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
from compile_hw import compiled_probs, load

tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "esnli_Born-ap_b2_s0_e20"
draws = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 1000
model, d = load(tag)
# the ledger keys of each run's two blocks; the earlier model's runs (attention squared again, arXiv v1-v2) are in the
# history of this repository (commit 7d94f41, "Version 3.1")
RUNS = {"esnli_Born-ap_b2_s0_e20": {       # 32 calibration circuits as deep as the overlaps per block
        "pair 3763": (3763, None, ["ae3d09c69431cc3d", "26df20031defaaa7"]),
        "pair 3765": (3765, None, ["a1d6be83e81a3267", "4327c6e5c51fae25"]),
        "pair 3763, aliens deleted": (3763, 5, ["c8b83b38235c0090", "f3abef9760911786"])}}[tag]
SAVE = f"results/{tag}_device_bootstrap.json"


def readings(keys):
    """{block: {(h, i, j, part): (P0 - P1, shots)}} from the ledger's notes and counts."""
    out = {}
    for b, k in enumerate(keys):
        rec = json.load(open(f"ibm_jobs/{k}/record.json"))
        counts = json.load(open(f"ibm_jobs/{k}/counts.json"))
        for c, cnt in zip(rec["circuits"], counts):
            m = re.match(r"block (\d) head (\d) q(\d+) k(\d+) (Re|Im|calibration)", c["note"])
            if m is None:                                   # a random-pair calibration of known overlap
                m = re.match(r"block (\d) random calibration (\d+) ideal ([-+.\d]+)$", c["note"])
                n = sum(cnt.values())
                out.setdefault(int(m[1]), {})[("random", int(m[2]), float(m[3]))] = \
                    ((cnt.get("0", 0) - cnt.get("1", 0)) / n, n)
                continue
            blk, h, i, j, part = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
            n = sum(cnt.values())
            out.setdefault(blk, {})[(h, i, j, part)] = ((cnt.get("0", 0) - cnt.get("1", 0)) / n, n)
    return out


def overlap_from(table, rng=None):
    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        H, L = q.shape[0], q.shape[1]
        re_, im_ = torch.zeros(1, H, L, L, dtype=torch.float64), torch.zeros(1, H, L, L, dtype=torch.float64)
        draw = lambda v, n: v if rng is None else 2 * rng.binomial(n, (1 + np.clip(v, -1, 1)) / 2) / n - 1
        cal = [(key[2], draw(v, n)) for key, (v, n) in table[block].items() if key[0] == "random"]
        off = 0.0
        if cal:                                   # as hw_attention.py: the readout offset fitted from the
            x, y = np.array(cal).T                # random-pair calibrations is taken off every overlap
            off = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0][1]
        for key, (v, n) in table[block].items():
            if key[0] == "random" or key[3] == "calibration":
                continue
            h, i, j, part = key
            scale = np.linalg.norm(q[h, i]) * np.linalg.norm(k[h, j])
            (re_ if part == "Re" else im_)[0, h, i, j] = (draw(v, n) - off) * scale
        return re_, im_
    return overlap


only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
out = {}
if only:
    import os
    if os.path.exists(SAVE):
        out = json.load(open(SAVE))
    elif os.path.exists("results/device_bootstrap_3763.json"):
        out = json.load(open("results/device_bootstrap_3763.json"))
for name, (idx, delete, keys) in RUNS.items():
    if only and name != only:
        continue
    ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
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
json.dump(out, open(SAVE, "w"), indent=1)
