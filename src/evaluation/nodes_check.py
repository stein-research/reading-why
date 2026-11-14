"""Is the exact rule's node count (ceil(degree / 2) = 100 for the Born Transformer) a loose bound? BEW with 30, 100
and 400 Gauss-Legendre nodes on the first ten test pairs (seed 0): the largest difference from 400 nodes.

python nodes_check.py [--tag esnli_Born-ap_b2_s0_e20]   # writes results/<tag>_nodes_check.json
"""
import json
import pathlib
import sys

import torch

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
from compile_hw import load
from evidence import evidence

tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "esnli_Born-ap_b2_s0_e20"
model, d = load(tag)
ids = torch.tensor(d["x_test"][:10])
with torch.no_grad():
    target = model(ids).argmax(1)
ev = {n: evidence(model, ids, target, n_nodes=n)[0].detach() for n in (30, 100, 400)}
res = dict(tag=tag, pairs=10, degree=int(model.degree()),
           max_diff_vs_400={n: float((ev[n] - ev[400]).abs().max()) for n in (30, 100)})
print(res)
json.dump(res, open(f"results/{tag}_nodes_check.json", "w"), indent=1)
