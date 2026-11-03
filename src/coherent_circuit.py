"""The Coherent-attn model as one qubit circuit with one measurement: build it, check it against the trained model,
count what it costs.

python coherent_circuit.py <tag> [--index 3765 3763] [--cost-only p ...]

The model's amplitude (models.CoherentAttn) is
    phi_b = sum_h sum_v R[b, (h, v)] sum_j (x_j^H M_h x_bar) (Wv_h x_j)_v,   M_h = Wk_h^H Wq_h,
with x_j the slot fields (words and the reference slot) and x_bar = sum_j x_j. The circuit:
- copy A: x_bar on 5 feature qubits; copy B: sum_j |j>|conj x_j> and copy C: sum_j |j>|x_j> on p position + 5
  feature qubits each (amplitude encoding of fields the computer prepared); a 2-qubit head register in |+>|+>;
- head-controlled block encodings of M_h on A and of Wv_h on C (one ancilla each; one common scale per family, so
  the heads keep their relative weight);
- a Bell projection of B's and A's feature registers (CNOT, H, post-select 0) gives x_j^H M_h x_bar;
- CNOT from B's to C's position register, post-select 0 (j = j'); H on B's, post-select 0 (the sum over j);
- R: a block encoding on (C's features, head, one pad qubit) with one ancilla, measuring 8 qubits; or, for
  models.CoherentAttnU, a brickwall of two-qubit unitaries on (C's features, head), measuring those 7.
Every post-selection and block encoding only scales the state, so P_g = (region sums) / (total) is the model's.
Noiseless statevector check (qiskit-aer) on short pairs; gate counts by transpiling each block on its own.
"""
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit_aer import AerSimulator

from compile_hw import load

args = sys.argv[1:]
tag = args[0]
idxs = [int(a) for a in args[args.index("--index") + 1:] if a.isdigit()] if "--index" in args else [3765, 3763]
cost_only = [int(a) for a in args[args.index("--cost-only") + 1:]] if "--cost-only" in args else None
# --gpu: simulate in the GPU venv (qiskit-aer-gpu + cuStateVec, no torch there) through sim_gpu.py
GPU_PY = os.environ.get("QML_GPU_PYTHON", sys.executable) if "--gpu" in args else None   # a Python with qiskit-aer-gpu
if cost_only:
    idxs = []

model, d = load(tag)                                    # float64, CPU
C = lambda p: torch.complex(p[0], p[1]).detach().numpy()
H, DV, D = model.heads, model.dv, model.d
# models.CoherentAttnU: R is a brickwall of two-qubit unitaries on (featC, head), no pad or ancilla;
# models.CoherentAttnB: U_M on (featA, head) and U on (featC, head) are brickwalls too, no block encoding at all
BRICK = hasattr(model, "gates")
ALL_BRICK = hasattr(model, "mgen")


def fields(ids):
    """Slot fields x_t (words, then the reference slot), as models.CoherentAttn.numerators builds them."""
    ids = torch.tensor(ids[None])
    amp = model.amplitudes(ids)
    L = ids.shape[1]
    seg = (torch.cumsum((ids == model.sep).long(), 1) > 0).long()
    e = model._c(model.emb)[ids] + model._c(model.pos)[:L] + model._c(model.seg)[seg]
    t = torch.arange(L, dtype=model.omega.dtype)
    e = e * torch.polar(torch.ones_like(t[:, None] * model.omega), t[:, None] * model.omega)
    x = (e * amp.unsqueeze(-1).to(e.dtype))[0]
    present = ids[0] != 0
    return torch.cat((x[present], model._c(model.ref)[None]), 0).detach().numpy()


def dilation(A):
    """Unitary [[A, sqrt(1 - A A^H)], [sqrt(1 - A^H A), -A^H]] for ||A|| <= 1 (one ancilla, the high bit)."""
    u, s, vh = np.linalg.svd(A)
    c = np.sqrt(np.clip(1 - s ** 2, 0, None))
    top = np.hstack([A, u @ np.diag(c) @ u.conj().T])
    bot = np.hstack([vh.conj().T @ np.diag(c) @ vh, -A.conj().T])
    U = np.vstack([top, bot])
    assert np.allclose(U @ U.conj().T, np.eye(len(U)), atol=1e-9)
    return U


def head_controlled(mats):
    """Block-diagonal over the head register (high bits) of each head's dilation; one common scale."""
    scale = max(np.linalg.norm(m, 2) for m in mats)
    blocks = [dilation(m / scale) for m in mats]
    n = len(blocks[0])
    U = np.zeros((n * len(blocks), n * len(blocks)), dtype=complex)
    for h, b in enumerate(blocks):
        U[h * n:(h + 1) * n, h * n:(h + 1) * n] = b
    return U


if not ALL_BRICK:
    Wq, Wk, Wv = C(model.wq), C(model.wk), C(model.wv)
    M = np.stack([Wk[h].conj().T @ Wq[h] for h in range(H)])                 # [H, D, D]
    UM = head_controlled(list(M))                                             # qubits [featA(5), ancM, head(2)]
    UV = head_controlled(list(Wv))                                            # qubits [featC(5), ancV, head(2)]
else:
    GATES_M = model.m_gates().detach().numpy()
if BRICK:
    GATES = model.gates().detach().numpy()                                    # [layers, 3, 4, 4]
else:
    Rpad = np.zeros((256, 256), dtype=complex)
    Rpad[:, :H * DV] = C(model.readout)
    UR = dilation(Rpad / np.linalg.norm(Rpad, 2))                              # qubits [featC, head, pad, ancR]
NQ0 = 17 if ALL_BRICK else 19 if BRICK else 21                               # qubits besides the slot registers


def brick(qc, gates, pairs, qubits):
    for l, layer in enumerate(pairs):
        for i, (a, b) in enumerate(layer):                                    # gate index b_a + 2 b_(a+1)
            qc.append(UnitaryGate(gates[l, i], check_input=False), [qubits[a], qubits[b]])


def readout(qc, qubits):
    """R on (featC, head[, pad, ancR]): the brickwall gate by gate, or the dense block encoding."""
    if BRICK:
        brick(qc, GATES, model.pairs, qubits)
    else:
        qc.append(UnitaryGate(UR, check_input=False), qubits)


def build(x, p, paired=True):
    """The circuit for slot fields x [n, D] with p position qubits; returns (circuit, register map).
    paired: prepare B and C together as sum_j |j>|conj x_j>|x_j> on one position register (21 + p qubits; 19 + p
    with the brickwall readout, 17 + p with every map a brickwall), the state the separate preparations reach
    after the j = j' post-selection; else B and C apart (one more p)."""
    n = len(x)
    assert n <= 2 ** p
    names = [("featC", 5), ("head", 2)] + ([] if BRICK else [("pad", 1), ("ancR", 1)]) + \
            ([] if paired else [("posC", p)]) + [("posB", p), ("featB", 5), ("featA", 5)] + \
            ([] if ALL_BRICK else [("ancM", 1), ("ancV", 1)])
    reg, at = {}, 0
    for nm, k in names:
        reg[nm] = list(range(at, at + k))
        at += k
    qc = QuantumCircuit(at)
    xb = np.zeros((2 ** p, D), dtype=complex)
    xb[:n] = x
    vb, vc, va = xb.conj().reshape(-1), xb.reshape(-1), x.sum(0)            # index = feature + 32 * slot
    # State preparation from |0>: cost() counts StatePreparation's gates block by block; the simulation starts
    # from the prepared state instead (set_statevector), since Aer's initialize on these registers stalled (CPU)
    # or ran out of memory (GPU).
    nz = lambda v: v / np.linalg.norm(v)

    def prep(v, qubits):
        m = np.arange(len(v))
        at = sum(((m >> i) & 1) << q for i, q in enumerate(qubits))
        full = np.zeros(2 ** qc.num_qubits, dtype=complex)
        full[at] = v
        qc.set_statevector(full)
    if paired:                                                               # index = c + 32 b + 1024 j
        vbc = np.einsum("jb,jc->jbc", xb.conj(), xb).reshape(-1)
        prep(np.kron(nz(va), nz(vbc)), reg["featC"] + reg["featB"] + reg["posB"] + reg["featA"])
    else:
        prep(np.kron(nz(va), np.kron(nz(vc), nz(vb))),
             reg["featB"] + reg["posB"] + reg["featC"] + reg["posC"] + reg["featA"])
    qc.h(reg["head"])
    if ALL_BRICK:
        brick(qc, GATES_M, model.pairs_m, reg["featA"] + reg["head"])         # m = U_M (x_bar (x) |++>)
    else:
        qc.append(UnitaryGate(UM, check_input=False), reg["featA"] + reg["ancM"] + reg["head"])
        qc.append(UnitaryGate(UV, check_input=False), reg["featC"] + reg["ancV"] + reg["head"])
    for b, a in zip(reg["featB"], reg["featA"]):                               # Bell projection B . A
        qc.cx(b, a)
    qc.h(reg["featB"])
    if not paired:
        for b, c in zip(reg["posB"], reg["posC"]):                             # j = j'
            qc.cx(b, c)
    qc.h(reg["posB"])                                                         # the sum over j
    readout(qc, reg["featC"] + reg["head"] + reg.get("pad", []) + reg.get("ancR", []))
    return qc, reg


def run(i, p=3):
    ids = d["x_test"][i]
    x = fields(ids)
    qc, reg = build(x, p)
    nq = qc.num_qubits
    zero = [q for nm in ("ancR", "posC", "posB", "featB", "featA", "ancM", "ancV") for q in reg.get(nm, [])]
    t0 = time.time()
    if GPU_PY:
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/c.pkl", "wb") as f:      # pickle: qpy failed to read the large prepared state
                pickle.dump(qc, f)
            subprocess.run([GPU_PY, "sim_gpu.py", f"{tmp}/c.pkl", f"{tmp}/out.npy", ",".join(map(str, zero))],
                           check=True)
            out = np.load(f"{tmp}/out.npy")
    else:
        qc.save_statevector()
        sim = AerSimulator(method="statevector", fusion_enable=False, max_parallel_threads=4)
        sv = np.asarray(sim.run(qc).result().get_statevector())
        psi = sv.reshape([2] * nq)                                            # axis k <-> qubit nq - 1 - k
        sel = [slice(None)] * nq
        for q in zero:
            sel[nq - 1 - q] = 0
        out = psi[tuple(sel)].reshape(-1)                                     # index = featC + 32 head + 128 pad
    secs = time.time() - t0
    prob = np.abs(out) ** 2
    success = float(prob.sum())
    # the separate preparations (21 + 2p qubits) pass j = j' with probability sum_j |x_j|^4 / |X|^4
    nrm = np.linalg.norm(x, axis=1)
    success_apart = success * float((nrm ** 4).sum() / (nrm ** 2).sum() ** 2)
    p_circ = (model.groups.numpy() @ prob) / success
    with torch.no_grad():
        p_model = model(torch.tensor(ids[None]))[0].numpy()
    words = " ".join(d["words"][t] for t in ids if t)
    return dict(index=i, words=words, gold=d["classes"][d["y_test"][i]], qubits=nq,
                P_circuit=p_circ.tolist(), P_model=p_model.tolist(), gap=float(np.abs(p_circ - p_model).max()),
                post_selection=success, post_selection_apart=success_apart,
                qubits_apart=nq + p, seconds=secs)


def block_cost(qc):
    t0 = time.time()
    tq = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    ops = tq.count_ops()
    return dict(cx=int(ops.get("cx", 0)), u=int(ops.get("u", 0)), depth=tq.depth(), seconds=time.time() - t0)


def cost(p):
    """Two-qubit gates block by block for p position qubits (fields drawn at random; counts do not depend on them)."""
    rng = np.random.default_rng(0)
    res = {}
    for nm, k in (("prep B/C (each)", p + 5), ("prep B+C paired", p + 10), ("prep A", 5)):
        v = rng.normal(size=2 ** k) + 1j * rng.normal(size=2 ** k)
        q = QuantumCircuit(k)
        q.append(StatePreparation(v / np.linalg.norm(v)), range(k))
        res[nm] = block_cost(q)
    if ALL_BRICK:
        q = QuantumCircuit(7)
        brick(q, GATES_M, model.pairs_m, list(range(7)))
        res["U_M"] = block_cost(q)
    else:
        for nm, U in (("head-controlled M", UM), ("head-controlled Wv", UV)):
            k = int(round(math.log2(len(U))))
            q = QuantumCircuit(k)
            q.append(UnitaryGate(U, check_input=False), range(k))
            res[nm] = block_cost(q)
    q = QuantumCircuit(7 if BRICK else 9)
    readout(q, list(range(q.num_qubits)))
    res["readout R"] = block_cost(q)
    res["projections (CNOT)"] = dict(cx=5 + p)
    rest = sum(v["cx"] for k, v in res.items() if not k.startswith("prep B"))
    return dict(p=p, qubits=NQ0 + 2 * p, blocks=res, cx_total=2 * res["prep B/C (each)"]["cx"] + rest,
                qubits_paired=NQ0 + p, cx_total_paired=res["prep B+C paired"]["cx"] + rest - p)


out = dict(tag=tag, runs=[run(i) for i in idxs])
for r in out["runs"]:
    print(f"{r['index']} ({r['gold']}) {r['words']}: P circuit {np.round(r['P_circuit'], 6).tolist()} model "
          f"{np.round(r['P_model'], 6).tolist()} gap {r['gap']:.1e}, post-selection {r['post_selection']:.2e} "
          f"({r['post_selection_apart']:.2e} with B, C apart on {r['qubits_apart']}), {r['qubits']} qubits, {r['seconds']:.0f} s", flush=True)
out["cost"] = [cost(p) for p in (cost_only or [3])]
for c in out["cost"]:
    print(f"p {c['p']}: {c['qubits']} qubits, CX total {c['cx_total']:,}; paired {c['qubits_paired']} qubits, "
          f"CX {c['cx_total_paired']:,}; " +
          ", ".join(f"{k} {v['cx']:,}" for k, v in c["blocks"].items()), flush=True)
with open(f"results/{tag}_circuit.json", "w") as f:
    json.dump(out, f, indent=1)
