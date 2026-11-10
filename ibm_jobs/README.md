# IBM hardware jobs

The record of every job the paper ran on IBM's **ibm_kingston** (Heron r2), so that every hardware number in the
paper can be checked from the raw counts, without an IBM account. Each folder is one job, named by a key over its
circuits, the backend and the shots; [`index.md`](index.md) lists them with their IBM job ids, times and quantum
seconds. Account details are removed.

## The six jobs

Only the attention overlaps ran on the device: every $\langle q_i|k_j\rangle$ of both blocks of the Born Transformer
(seed 0), read by a five-qubit Hadamard test, 2,000 shots per circuit. Each block is one job; block 2's circuits are
built from block 1's measured overlaps.

| folder | run | block | circuits | quantum s |
|---|---|---|---|---|
| [`ae3d09c69431cc3d`](ae3d09c69431cc3d/record.json) | pair 3763, "men play hockey . / aliens play" (contradiction) | 1 | 288 | 155 |
| [`26df20031defaaa7`](26df20031defaaa7/record.json) | pair 3763 | 2 | 288 | 155 |
| [`a1d6be83e81a3267`](a1d6be83e81a3267/record.json) | pair 3765, "men play hockey . / men play" (entailment) | 1 | 288 | 155 |
| [`4327c6e5c51fae25`](4327c6e5c51fae25/record.json) | pair 3765 | 2 | 288 | 155 |
| [`c8b83b38235c0090`](c8b83b38235c0090/record.json) | pair 3763 without "aliens" | 1 | 210 | 114 |
| [`f3abef9760911786`](f3abef9760911786/record.json) | pair 3763 without "aliens" | 2 | 210 | 114 |

A block of a pair holds 256 overlap circuits (the real and imaginary parts of 128 overlaps) and 32 calibration
circuits as deep as the overlaps (Hadamard tests of random state pairs of known overlap, from which the block's
visibility and readout offset are fitted). A block of the deletion holds 196 overlap circuits and 14 shallow row
calibrations, which are not used: the deletion asks only for the answer. In all, 848 quantum seconds.

What they give (the paper's Section 5, Tables 4 and 5):

| run | answer: device / noiseless | each word's evidence with the device's row sums |
|---|---|---|
| pair 3763 | contradiction 0.901 / 0.922 | correlation 0.962 with the exact evidence |
| pair 3763 without "aliens" | entailment 0.857 / 0.912 | (answer only) |
| pair 3765 | entailment 0.925 / 0.905 | correlation 0.997 |

## What a folder holds

| file | what |
|---|---|
| `record.json` | job id, backend, status, shots, times, quantum seconds, IBM's own metrics, the qubits each circuit ran on, the command line, the code and package versions, the input model file with its SHA-256, and a note for every circuit (which overlap, real or imaginary part, or which calibration and its ideal value) |
| `circuits.qpy` | the circuits as built (Qiskit QPY) |
| `transpiled.qpy` | the circuits as sent to ibm_kingston |
| `counts.json` | the raw counts of every circuit, in the order of `record.json`'s `circuits` |
| `properties.json` | the backend's calibration when the job was sent (T1, T2, gate and readout errors) |

A circuit's reading is `(counts["0"] - counts["1"]) / 2000`: the real or imaginary part of an overlap, times the
device's visibility.

## Check the numbers yourself

No account and no network are needed; nothing is sent. From the repository's top folder:

```bash
python src/ibm_jobs.py list                     # the job list (rewrites index.md)
python src/device_overlaps.py                   # readings against their ideal values, per job
python src/device_bootstrap.py                  # the answers and their shot-noise intervals
python src/evidence_bootstrap.py --index 3765   # the evidence on the device (and --index 3763)
python src/hw_attention.py esnli_Born-ap_b2_s0_e20 --index 3765 --shots 2000 --device ibm_kingston --cal-random 32
```

The last command rebuilds the whole pass from the model and takes each block's counts from this folder ("already
done ... counts from the ledger, nothing sent"). Everything, including how to run the jobs again with your own
account: [`docs/check-and-repeat-ibm.md`](../docs/check-and-repeat-ibm.md).

## Earlier jobs

The paper's earlier versions (arXiv v1-v2) used a model whose attention weight was squared again, $|\langle
q|k\rangle|^4$. Its jobs on the same device are at the tag [`v3.1`](https://github.com/stein-research/reading-why/tree/v3.1/ibm_jobs)
of this repository and in its Zenodo archive of that version.
