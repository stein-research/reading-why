"""Which row sums the hybrid evidence holds, and the Hadamard-test deletion run's evidence (referee round 6,
2026-11-12). From the ledger's counts for pairs 3763, 3765 and 3763 without "aliens": the device's answer and the
hybrid evidence with each block's slope from its depth-matched calibrations (cal), from its overlaps against their
known ideal values (ov), and, for the deletion run, from its 14 shallow self-overlap calibrations (self); the block-2
vectors the device read against the model's own (their row sums), and the evidence with the measured row sums held
as they are, in model units (ABSOLUTE).
python device_rows.py   (CPU, a few minutes) -> results/device_rows.json (its printed lines)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)

import json, math, re, sys
import numpy as np, torch
import pathlib

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import compile_hw
from compile_hw import compiled_probs, load
_vm = compile_hw.vmax
VM = []
def _rec(x, amp):
    v = _vm(x, amp); VM.append(float(v)); return v
compile_hw.vmax = _rec
from evidence import _score_fn, evidence, nodes01
from models import Frozen
tag = "esnli_Born-ap_b2_s0_e20"
torch.set_num_threads(4)
model, d = load(tag)
C = d["classes"]
RUNS = {"3763": (3763, None, ["ae3d09c69431cc3d", "26df20031defaaa7"]), "3765": (3765, None, ["a1d6be83e81a3267", "4327c6e5c51fae25"]),
        "3763-aliens": (3763, 5, ["c8b83b38235c0090", "f3abef9760911786"])}
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
rng = np.random.default_rng(1)
for name, (idx, dele, keys) in RUNS.items():
    ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
    if dele is not None: ids[0, dele] = 0
    tok = np.where(ids[0].numpy() != 0)[0]
    present = list(tok) + [ids.shape[1]]
    L = ids.shape[1] + 1
    store = {}
    def make(mode):
        def overlap(qi, kj, block):
            q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
            rec = json.load(open(f"ibm_jobs/{keys[block]}/record.json"))
            cnt = json.load(open(f"ibm_jobs/{keys[block]}/counts.json"))
            H = q.shape[0]
            re_, im_ = np.zeros((H, L, L)), np.zeros((H, L, L)); cal = []; selfc = {}
            ideal, meas = [], []
            for c, n in zip(rec["circuits"], cnt):
                sh = sum(n.values()); p = (n.get("0", 0) - n.get("1", 0)) / sh
                m = re.match(r"block (\d) head (\d) q(\d+) k(\d+) (Re|Im|calibration)$", c["note"])
                if m:
                    h, i, j = int(m[2]), int(m[3]), int(m[4])
                    if m[5] == "calibration": selfc[(h, i)] = p; continue
                    nq, nk = np.linalg.norm(q[h, i]), np.linalg.norm(k[h, j])
                    ov = np.vdot(q[h, i], k[h, j]) / (nq * nk)
                    ideal.append(ov.imag if m[5] == "Im" else ov.real); meas.append(p)
                    (im_ if m[5] == "Im" else re_)[h, i, j] = p
                    continue
                m = re.match(r"block (\d) random calibration \d+ ideal ([-+.\d]+)$", c["note"]); cal.append((float(m[2]), p))
            ideal, meas = np.array(ideal), np.array(meas)
            s_ov, o_ov = np.polyfit(ideal, meas, 1)
            if cal:
                x, y = np.array(cal).T; s_cal, o_cal = np.polyfit(x, y, 1)
            else: s_cal, o_cal = float("nan"), 0.0
            nq = np.linalg.norm(q, axis=-1); nk = np.linalg.norm(k, axis=-1)
            scale = nq[:, :, None] * nk[:, None, :]
            s, o = 1.0, 0.0
            if mode == "cal": s, o = s_cal, o_cal
            elif mode == "ov": s, o = s_ov, o_ov
            eta = np.full((H, L), s)
            if mode == "self":
                o = 0.0
                for (h, i), p in selfc.items(): eta[h, i] = max(p, 1e-3)
            R, I = (re_ - o) * scale, (im_ - o) * scale
            ovx = np.einsum("hie,hje->hij", q.conj(), k)
            keep = np.zeros(L, bool); keep[present] = True
            s_ex = (np.abs(ovx) ** 2 * keep).sum(-1); s_hw = ((R ** 2 + I ** 2) / eta[:, :, None] ** 2 * keep).sum(-1)
            rows = np.where(s_ex > 0, s_hw / np.maximum(s_ex, 1e-300), 1.0)
            store.setdefault(mode, {})[block] = dict(s_hw=s_hw, s_ex=s_ex, rows=torch.tensor(rows), s_cal=s_cal, o_cal=o_cal, s_ov=s_ov, o_ov=o_ov,
                 corr=float(np.corrcoef(ideal, meas)[0, 1]), resid=float(np.std(meas - s_ov * ideal - o_ov)),
                 selfmean=float(np.mean(list(selfc.values()))) if selfc else None)
            return torch.tensor(R)[None], torch.tensor(I)[None]
        return overlap
    with torch.no_grad():
        pe = compiled_probs(model, ids)[0]
    target = pe.argmax(1)
    ex = evidence(model, ids, target)[0][0, tok].detach().numpy()
    words = [d["words"][t] for t in ids[0][tok].tolist()]
    print(f"\n=== {name}: {' '.join(words)} exact {C[int(target)]} {pe[0, target].item():.3f}")
    print(" exact BEW", np.round(ex, 3), "rho=sum|e|/|sum e|", round(float(np.abs(ex).sum() / abs(ex.sum())), 3))
    modes = ["ov", "self"] if dele is not None else ["cal", "ov"]
    for mode in modes:
        VM.clear()
        with torch.no_grad(): ph = compiled_probs(model, ids, overlap=make(mode))[0]
        vm_dev = list(VM)
        rows = [store[mode][b]["rows"] for b in (0, 1)]
        ev, full = hw_ev(ids, target, rows, tok)
        st = store[mode]
        logsd = [float(np.log(r[:, present].numpy()).std()) for r in rows]
        print(f" [{mode}] device P {ph[0, target].item():.3f} answer {C[int(ph.argmax())]} | slopes cal {st[0]['s_cal']:.3f},{st[1]['s_cal']:.3f} ov {st[0]['s_ov']:.3f},{st[1]['s_ov']:.3f} self {st[0]['selfmean']},{st[1]['selfmean']} | ov corr {st[0]['corr']:.3f},{st[1]['corr']:.3f}")
        print(f"   hybrid BEW {np.round(ev, 3)} corr {np.corrcoef(ex, ev)[0, 1]:.3f} top exact {words[int(ex.argmax())]} dev {words[int(ev.argmax())]} l1/sum {np.abs(ev - ex).sum() / abs(ex.sum()):.3f} P~ {full + float(model.share[target]):.3f} row logsd {np.round(logsd, 3)}")
        dev_rows = rows
        if mode == modes[0]: main_mode = mode
    # absolute: the row sums the device measured, in the model's units, held as they are
    nl = {}
    def exact_hook(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        ovx = np.einsum("hie,hje->hij", q.conj(), k)
        keep = np.zeros(L, bool); keep[present] = True
        nl[block] = (np.abs(ovx) ** 2 * keep).sum(-1)
        return torch.tensor(ovx.real)[None], torch.tensor(ovx.imag)[None]
    VM.clear()
    with torch.no_grad(): pn = compiled_probs(model, ids, overlap=exact_hook)[0]
    vm_nl = list(VM)
    assert (pn - pe).abs().max() < 1e-9
    kap = ((3 + vm_nl[1] ** 2) / (3 + vm_dev[1] ** 2)) ** 2          # (scale_dev / scale_nl)^4 entering block 2
    ab = []
    for b in (0, 1):
        st = store[main_mode][b]; kb = 1.0 if b == 0 else kap
        r = np.ones_like(nl[b]); r[:, present] = st["s_hw"][:, present] / (nl[b][:, present] * kb)
        ab.append(torch.tensor(r))
        tv = st["s_ex"][:, present] / (nl[b][:, present] * kb)
        print(f"   block {b+1}: row sums of the vectors the device read / the model's own row sums: mean {tv.mean():.3f} range {tv.min():.2f}-{tv.max():.2f} log sd {np.log(tv).std():.3f}; measured(calibrated)/model's: log sd {np.log(r[:, present]).std():.3f}, range {r[:, present].min():.2f}-{r[:, present].max():.2f}")
    ev, full = hw_ev(ids, target, ab, tok)
    print(f" ABSOLUTE (held row sums = the calibrated row sums the device measured, model units): corr {np.corrcoef(ex, ev)[0, 1]:.3f} top {words[int(ev.argmax())]} l1/sum {np.abs(ev - ex).sum() / abs(ex.sum()):.3f} BEW {np.round(ev, 3)} P~ {full + float(model.share[target]):.3f}")

_j.dump(dict(script="device_rows.py", lines=_LINES), open("results/device_rows.json", "w"), indent=1)
