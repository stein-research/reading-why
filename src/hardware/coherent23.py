"""The coherent layer's 23-qubit circuit as counted (referee round 6, 2026-11-12): B and C prepared apart, every
state prepared by gates from |0...0>, simulated and compared with the trained model, with its pass rate. (coherent_circuit.py
simulates the paired 20-qubit form started from the prepared state.)  python coherent23.py esnli_Coherent-attn-b12-24_b1_s0   (CPU, ~6 min per pair)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)

import os, sys, time, numpy as np, torch
REPO = os.path.dirname(os.path.abspath(__file__))      # src/hardware/, beside coherent_circuit.py
tag = sys.argv[1]
sys.argv = ["coherent_circuit.py", tag]
src = open(f"{REPO}/coherent_circuit.py").read().split("\nout = dict(tag=tag")[0]
exec(compile(src, "coherent_circuit.py", "exec"))
from qiskit.quantum_info import Statevector
def build_gates(x, p):
    n = len(x)
    names = [("featC", 5), ("head", 2), ("posC", p), ("posB", p), ("featB", 5), ("featA", 5)]
    reg, at = {}, 0
    for nm, k in names: reg[nm] = list(range(at, at + k)); at += k
    qc = QuantumCircuit(at)
    xb = np.zeros((2 ** p, D), dtype=complex); xb[:n] = x
    nz = lambda v: v / np.linalg.norm(v)
    # index = feature + 32 * slot  -> qubits feat (low) then pos
    qc.append(StatePreparation(nz(xb.conj().reshape(-1))), reg["featB"] + reg["posB"])
    qc.append(StatePreparation(nz(xb.reshape(-1))), reg["featC"] + reg["posC"])
    qc.append(StatePreparation(nz(x.sum(0))), reg["featA"])
    qc.h(reg["head"])
    brick(qc, GATES_M, model.pairs_m, reg["featA"] + reg["head"])
    for b, a in zip(reg["featB"], reg["featA"]): qc.cx(b, a)
    qc.h(reg["featB"])
    for b, c in zip(reg["posB"], reg["posC"]): qc.cx(b, c)
    qc.h(reg["posB"])
    brick(qc, GATES, model.pairs, reg["featC"] + reg["head"])
    return qc, reg
import os
IDX = [int(a) for a in os.environ.get("IDX", "3765 3763").split()]
for i in IDX:
    ids = d["x_test"][i]; x = fields(ids)
    n = len(x)
    if n > 8: print(i, "needs more than 8 slots:", n); continue
    qc, reg = build_gates(x, 3)
    t0 = time.time()
    tq = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    ops = tq.count_ops()
    sv = np.asarray(Statevector.from_instruction(tq).data)
    nq = tq.num_qubits; psi = sv.reshape([2] * nq)
    zero = [q for nm in ("posC", "posB", "featB", "featA") for q in reg[nm]]
    sel = [slice(None)] * nq
    for q in zero: sel[nq - 1 - q] = 0
    out = psi[tuple(sel)].reshape(-1); prob = np.abs(out) ** 2; succ = prob.sum()
    pc = model.groups.numpy() @ prob / succ
    with torch.no_grad(): pm = model(torch.tensor(ids[None]))[0].numpy()
    # the paper's closed form, Eq. (pass)
    nrm = np.linalg.norm(x, axis=1)
    words = " ".join(d["words"][t] for t in ids if t)
    print(f"{i} [{words}] {nq} qubits, CX {ops.get('cx', 0)}, u {ops.get('u', 0)}: P circuit {np.round(pc, 6)} model {np.round(pm, 6)} gap {np.abs(pc - pm).max():.1e}; pass rate simulated {succ:.3e} ({time.time() - t0:.0f} s)", flush=True)

_j.dump(dict(script="coherent23.py", lines=_LINES), open("results/coherent23.json", "w"), indent=1)
