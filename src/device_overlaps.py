"""The overlaps read on the device against their ideal values, from the ledger alone (no account, nothing sent):
for each job, the correlation and the straight-line fit (slope = the visibility, intercept = the readout offset) of
the measured P0 - P1 against the ideal value of the recorded circuit (its statevector).

python device_overlaps.py [job keys or ids ...]     # default: every job of ibm_jobs/
"""
import json
import pathlib
import sys

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Statevector

LEDGER = pathlib.Path(__file__).resolve().parent / "ibm_jobs"
if not LEDGER.exists():
    LEDGER = pathlib.Path(__file__).resolve().parent.parent / "ibm_jobs"
wanted = sys.argv[1:]
for folder in sorted(LEDGER.iterdir()):
    rec_path = folder / "record.json"
    if not rec_path.exists():
        continue
    rec = json.loads(rec_path.read_text())
    if rec["status"] != "done" or (wanted and folder.name not in wanted and rec.get("job_id") not in wanted):
        continue
    counts = json.loads((folder / "counts.json").read_text())
    with open(folder / "circuits.qpy", "rb") as f:
        circs = qpy.load(f)
    rows = {"overlap": [], "self-overlap calibration": [], "random calibration": []}
    for c, cnt, info in zip(circs, counts, rec["circuits"]):
        p = Statevector(c.remove_final_measurements(inplace=False)).probabilities([c.num_qubits - 1])
        kind = ("random calibration" if "random calibration" in info["note"] else
                "self-overlap calibration" if info["note"].endswith("calibration") else "overlap")
        rows[kind].append((p[0] - p[1], (cnt.get("0", 0) - cnt.get("1", 0)) / sum(cnt.values())))
    print(f"{folder.name} {rec['job_id']}  {rec['purpose'].split('Hadamard tests')[-1].strip(', ') or 'all'}")
    for kind, v in rows.items():
        if not v:
            continue
        x, y = np.array(v).T
        if np.ptp(x) < 1e-9:                                   # every ideal value the same: the mean only
            print(f"  {kind:25s} n {len(x):3d}  ideal {x[0]:.3f}, measured mean {y.mean():.3f}")
            continue
        slope, off = np.polyfit(x, y, 1)
        print(f"  {kind:25s} n {len(x):3d}  corr {np.corrcoef(x, y)[0, 1]:.3f}  slope {slope:.3f}  offset {off:+.4f}")
