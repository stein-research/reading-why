"""Run one pickled circuit's statevector on the GPU (qiskit-aer-gpu, cuStateVec) and save the amplitudes left when the
listed qubits are post-selected to 0 (the low qubits not listed, in order). Runs in a Python with qiskit-aer-gpu (QML_GPU_PYTHON; torch not needed).

python sim_gpu.py <circuit.pkl> <out.npy> <q,q,...>
"""
import sys
import time

import numpy as np
import pickle
from qiskit_aer import AerSimulator

src, dst, zero = sys.argv[1], sys.argv[2], [int(q) for q in sys.argv[3].split(",")]
with open(src, "rb") as f:
    qc = pickle.load(f)                                                      # same qiskit (2.6.1) in both venvs
qc.save_statevector()
t0 = time.time()
r = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=True).run(qc).result()
sv = np.asarray(r.get_statevector())
nq = qc.num_qubits
sel = [slice(None)] * nq
for q in zero:
    sel[nq - 1 - q] = 0                                                       # axis k <-> qubit nq - 1 - k
np.save(dst, sv.reshape([2] * nq)[tuple(sel)].reshape(-1))
print(f"  GPU {nq} qubits, {time.time() - t0:.1f} s", flush=True)
