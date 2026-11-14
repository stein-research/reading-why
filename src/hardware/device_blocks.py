"""The answer checks block by block (referee round 6, 2026-11-12): on the three device inputs, uniform attention and
random overlaps (present slots only) in block 1 alone, block 2 alone, and both, for the compiled Born Transformer
(seed 0): the answer and how often random overlaps keep the model's answer.
python device_blocks.py   (CPU, a few minutes) -> results/device_blocks.json (its printed lines)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)

import numpy as np, torch
import sys
import pathlib

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
from compile_hw import compiled_probs, load
torch.set_num_threads(3)
model, d = load("esnli_Born-ap_b2_s0_e20"); C = d["classes"]
rng = np.random.default_rng(0)
for name, idx, dele in (("3763", 3763, None), ("3765", 3765, None), ("3763 without aliens", 3763, 5)):
    ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
    if dele is not None: ids[0, dele] = 0
    pres = torch.tensor(np.concatenate([ids[0].numpy() != 0, [True]]))
    mask = (pres[None, None, :, None] & pres[None, None, None, :])
    def hook(which, kind):
        def f(qi, kj, block):
            ov = (qi.conj() * kj).sum(-1)
            if block in which:
                if kind == "uniform": return mask.to(ov.real.dtype) * torch.ones_like(ov.real), torch.zeros_like(ov.real)
                return mask * torch.tensor(rng.standard_normal(ov.shape)), mask * torch.tensor(rng.standard_normal(ov.shape))
            return ov.real, ov.imag
        return f
    with torch.no_grad():
        pe = compiled_probs(model, ids)[0][0]; t = int(pe.argmax())
        out = f"{name}: exact {C[t]} {pe[t]:.3f}"
        for which, nm in (((0,), "blk1"), ((1,), "blk2"), ((0, 1), "both")):
            pu = compiled_probs(model, ids, overlap=hook(which, "uniform"))[0][0]
            same = np.mean([int(compiled_probs(model, ids, overlap=hook(which, "random"))[0][0].argmax()) == t for _ in range(200)])
            out += f" | uniform {nm}: {C[int(pu.argmax())]} P(answer) {pu[t]:.3f}; random {nm} keeps answer {same:.2f}"
    print(out, flush=True)

_j.dump(dict(script="device_blocks.py", lines=_LINES), open("results/device_blocks.json", "w"), indent=1)
