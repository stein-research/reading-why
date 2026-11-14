"""Softmax baseline: integrated gradients with 256, 1024 and 4096 nodes on the first 200 test pairs - completeness
gap and top-3 words against 256 nodes. Writes results/softmax_ig_nodes_s0.json."""
import sys, time, json

import numpy as np, torch
import pathlib

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
from compile_hw import load
from evidence import evidence
m, d = load("esnli_Softmax_b2_s0_e20")
ids = torch.tensor(d["x_test"][:200])
with torch.no_grad():
    tgt = m(ids).argmax(1)
out, ref = {}, None
for n in (256, 1024, 4096):
    t = time.time()
    ev, full, zero = evidence(m, ids, tgt, n_nodes=n)
    ev = ev.detach().numpy(); gap = np.abs(ev.sum(1) - (full - zero).detach().numpy())
    top = [set(np.argsort(-e)[:3]) for e in ev]
    if ref is None: ref = top
    agree = np.mean([len(a & b) / 3 for a, b in zip(top, ref)])
    out[n] = dict(gap_median=float(np.median(gap)), gap_max=float(gap.max()), top3_same_as_256=float(agree), secs=time.time() - t)
    print(n, out[n], flush=True)
json.dump(out, open("results/softmax_ig_nodes_s0.json", "w"), indent=1)
