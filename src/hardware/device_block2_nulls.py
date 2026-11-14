"""What the device evidence check tests, block by block (referee round 7, 2026-11-12): on pairs 3763 and 3765 the
hybrid evidence with the device's row sums in one block and, in the other, exact row sums, row sums left uncorrected
for visibility, a constant normalised overlap (no device information, scale matched), Haar-random or permuted
overlaps, or log-normal row errors; correlation and l1 error against the exact evidence.
python device_block2_nulls.py   (CPU, ~15 min) -> results/device_block2_nulls.json (printed lines)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)
import json, numpy as np, torch
import sys
import pathlib

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen
torch.set_num_threads(4)
TAG = "esnli_Born-ap_b2_s0_e20"
model, d = load(TAG)
class HwFrozen(Frozen):
    def __init__(s, rows): super().__init__(); s.rows, s.blk = rows, 0
    def __call__(s, x):
        if not s.replay and x.dim() == 4:
            x = x * s.rows[s.blk][None, :, :, None].to(x.dtype); s.blk += 1
        return super().__call__(x)
def hw_ev(ids, target, rows, tok):
    x0, f = _score_fn(model, ids); x0 = x0.detach(); fz = HwFrozen(rows)
    with torch.no_grad(): full = f(x0, fz).gather(1, target[:, None])[0, 0].item()
    tot = torch.zeros_like(x0)
    for t, w in zip(*nodes01(40)):
        x = (float(t) * x0).requires_grad_(True)
        (g,) = torch.autograd.grad(f(x, fz.start_replay()).gather(1, target[:, None]).sum(), x); tot += float(w) * g
    return (x0 * tot).detach()[0, tok].numpy(), full
rng = np.random.default_rng(3)
for idx in (3763, 3765):
    r = json.load(open(f"results/{TAG}_device_ibm_kingston_2000_i{idx}_cal32.json"))
    dev_rows = [np.array(b) for b in r["device_run"]["eta"]]; slopes = [c["slope"] for c in r["device_run"]["calibration"]]
    ids = torch.tensor(d["x_test"][idx:idx + 1]); tok = np.where(ids[0].numpy() != 0)[0]; present = list(tok) + [ids.shape[1]]; L = ids.shape[1] + 1
    keep = np.zeros(L, bool); keep[present] = True
    QK = {}
    def hook(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy(); QK[block] = (q, k)
        o = np.einsum("hie,hje->hij", q.conj(), k); return torch.tensor(o.real)[None], torch.tensor(o.imag)[None]
    with torch.no_grad(): pe = compiled_probs(model, ids, overlap=hook)[0]
    target = pe.argmax(1); ex = evidence(model, ids, target)[0][0, tok].detach().numpy()
    one = np.ones((2, L))
    def rep(rows):
        ev, full = hw_ev(ids, target, [torch.tensor(x) for x in rows], tok)
        return np.corrcoef(ex, ev)[0, 1], np.abs(ev - ex).sum() / abs(ex.sum()), int(ev.argmax()) == int(ex.argmax()), full + float(model.share[target])
    def null_rows(b, kind):
        q, k = QK[b]; nq = np.linalg.norm(q, axis=-1); nk = np.linalg.norm(k, axis=-1); n2 = (nq[:, :, None] * nk[:, None, :]) ** 2
        o = np.einsum("hie,hje->hij", q.conj(), k); s_ex = (np.abs(o) ** 2 * keep).sum(-1)
        if kind == "const": rr = np.full_like(n2, 1 / 16)
        elif kind == "haar": rr = (rng.standard_normal(n2.shape) ** 2 + rng.standard_normal(n2.shape) ** 2) / 32
        if kind == "perm":
            x = np.ones_like(s_ex)
            for h in range(2):
                p = rng.permutation(present); x[h, present] = s_ex[h, p] / s_ex[h, present]
            return x
        s_hw = (rr * n2 * keep).sum(-1); x = np.ones_like(s_ex); x[:, present] = s_hw[:, present] / s_ex[:, present]
        x[:, present] /= np.exp(np.log(x[:, present]).mean()); return x
    f = lambda t: f"corr {t[0]:.3f} l1 {t[1]:.3f} top {'same' if t[2] else 'DIFF'} P~ {t[3]:.3f}"
    print(f"\n=== pair {idx}  (device slopes {np.round(slopes,3)})")
    print("  device both blocks           :", f(rep(dev_rows)))
    print("  device block1, EXACT block2  :", f(rep([dev_rows[0], one])))
    print("  EXACT block1, device block2  :", f(rep([one, dev_rows[1]])))
    print("  device b1, block2 UNCORRECTED for visibility (x slope^2):", f(rep([dev_rows[0], dev_rows[1] * slopes[1] ** 2])))
    print("  block1 UNCORRECTED, device b2:", f(rep([dev_rows[0] * slopes[0] ** 2, dev_rows[1]])))
    c = rep([dev_rows[0], null_rows(1, "const")]); print("  device b1, block2 = constant normalised overlap (no device information, scale matched):", f(c))
    for kind in ("haar", "perm"):
        cs = np.array([rep([dev_rows[0], null_rows(1, kind)])[:3] for _ in range(60)])
        dv = rep(dev_rows)[0]
        print(f"  device b1, block2 = {kind}: corr median {np.median(cs[:,0]):.3f} p5 {np.percentile(cs[:,0],5):.3f} p95 {np.percentile(cs[:,0],95):.3f}; share >= device's {np.mean(cs[:,0] >= dv):.2f}; l1 median {np.median(cs[:,1]):.3f}; top same {cs[:,2].mean():.2f}")
    for kind in ("haar", "perm"):
        cs = np.array([rep([null_rows(0, kind), dev_rows[1]])[:3] for _ in range(60)])
        print(f"  block1 = {kind}, device b2: corr median {np.median(cs[:,0]):.3f} p5 {np.percentile(cs[:,0],5):.3f} p95 {np.percentile(cs[:,0],95):.3f}; l1 median {np.median(cs[:,1]):.3f}")
    for sd in (0.3, 0.6, 1.0):
        cs = []
        for _ in range(60):
            x = np.ones((2, L)); x[:, present] = np.exp(rng.normal(0, sd, (2, len(present)))); cs.append(rep([dev_rows[0], x])[0])
        print(f"  device b1, block2 rows x lognormal(sd {sd}): corr median {np.median(cs):.3f} p5 {np.percentile(cs,5):.3f}; share >= device's {np.mean(np.array(cs) >= rep(dev_rows)[0]):.2f}")

_j.dump(dict(script="device_block2_nulls.py", lines=_LINES), open("results/device_block2_nulls.json", "w"), indent=1)
