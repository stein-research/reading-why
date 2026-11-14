# Checking and repeating the IBM jobs

Every hardware number in the paper comes from jobs run on IBM's ibm_kingston, and every job is recorded in
[`ibm_jobs/`](../ibm_jobs/): the circuits as built and as sent, the backend's calibration at the time and the raw
counts (what each file holds: [`ibm_jobs/README.md`](../ibm_jobs/README.md); the list of jobs: [`ibm_jobs/index.md`](../ibm_jobs/index.md)). The jobs of the paper's earlier version
(attention squared again, arXiv v1-v2) are in this repository's history, at commit 7d94f41 ("Version 3.1"). This page shows three things, from the least to the most work:

1. check the paper's hardware numbers from those records, with no IBM account;
2. run the same circuits again on IBM hardware with your own account;
3. send the recorded circuits with plain Qiskit, without our scripts.

Install first as in [`run-locally.md`](run-locally.md) (the full `requirements.txt`). Every command runs from the root
of the clone. `TAG` is the model the hardware jobs used (the Born Transformer, seed 0, 20 epochs):

```bash
TAG=esnli_Born-ap_b2_s0_e20
```

The scripts that rebuild a pass need e-SNLI's test split, which `src/core/esnli.py` downloads into `data/esnli/` the first
time (see [`data/README.md`](../data/README.md)).

## 1. Check the numbers, no account

These read only the files in the repository and send nothing.

| in the paper | command | prints | time |
|---|---|---|---|
| quantum seconds of every job (Table 13) | `python src/hardware/ibm_jobs.py list` | the job list, rewritten into `ibm_jobs/index.md` | seconds |
| the overlaps against their ideal values (5): correlation 0.992-0.993 over the 256 readings of each block; visibility 0.77 / 0.67 (pair 3763) and 0.81 / 0.79 (pair 3765) as the slope fitted to the 32 calibrations of each block | `python src/hardware/device_overlaps.py` (or name job keys: `... ae3d09c69431cc3d 26df20031defaaa7`) | per job and kind of circuit: correlation, slope (the visibility), offset | a minute |
| the device runs of Table 4 (answers of pairs 3763 and 3765, and of 3763 without "aliens") and their 95 % intervals | `python src/hardware/device_bootstrap.py` (`--only "pair 3763"` for one run) | each run's P on the device, its interval, and the share of redraws that keep the answer | about 15 min a run |
| the evidence on the device (5, Table 6, Fig. 1b): pair 3765 correlation 0.997, interval 0.988-0.999, every sign and the same top word in every redraw; pair 3763 0.962, interval 0.880-0.991, every sign in 99 % of redraws | `python src/hardware/evidence_bootstrap.py --index 3765` (and `--index 3763`; 500 redraws, `--draws 20` to see the device values in a minute) | the device's evidence, each word's interval, the correlation and its interval, the fitted calibrations | about an hour |
| which single-word deletion changes the answer on the device pairs (5): only "aliens" on pair 3763, for seed 0 (the device model) | `python src/evaluation/deletion_words.py` | every deletion of both pairs in the three seeds, with each pair's exact evidence | minutes |
| compute-uncompute against the Hadamard tests on the same 128 Born weights (Table 7: MAE 0.021 for the Hadamard tests, 0.035 for compute-uncompute; correlation 0.986 and 0.979) | `python src/hardware/hw_cu_compare.py` | each method's calibrated weights against the ideal: MAE, RMSE, bias, correlation | a minute |
| the two compute-uncompute runs (Table 7): pair 3765, answer 0.877 and evidence correlation 0.971; 3763 without "aliens", answer 0.865 and evidence correlation 0.48 | `python src/hardware/cu_analysis.py` (`--draws 20 --ev-draws 10 --null-draws 5` to see the device values in a few minutes) | per block the calibration slope, offset and scatter against shot noise, the CZ counts; per run the answer, its 95 % interval, the evidence and its interval | minutes |

The bootstraps read the counts in `ibm_jobs/` directly. To rebuild a whole pass from the raw counts
with the script that sent it, run its device command without `--submit`: it rebuilds every circuit from the model,
finds each job in `ibm_jobs/` by its key, takes the counts from there, and logs in nowhere and sends nothing (it
prints `already done ... counts from the ledger, nothing sent` for each block):

```bash
python src/hardware/hw_attention.py $TAG --index 3763 --shots 2000 --device ibm_kingston --cal-random 32
python src/hardware/hw_attention.py $TAG --index 3765 --shots 2000 --device ibm_kingston --cal-random 32
python src/hardware/hw_attention.py $TAG --index 3763 --shots 2000 --device ibm_kingston --delete 5
```

The two compute-uncompute passes (`--cu`: four qubits, one circuit per overlap) rebuild the same way:

```bash
python src/hardware/hw_attention.py $TAG --index 3765 --shots 2000 --device ibm_kingston --cal-random 32 --cu
python src/hardware/hw_attention.py $TAG --index 3763 --shots 2000 --device ibm_kingston --cal-random 32 --cu --delete 5
```

They print pair 3765: entailment 0.877 against 0.905 exact, evidence correlation 0.971; and 3763 without "aliens":
entailment 0.865 against 0.912, evidence correlation 0.48 (Table 7).

The deletion run asks only for the answer: the evidence it also prints comes from shallow self-overlap calibrations,
which the paper does not use. Block 2's circuits are built from block 1's measured overlaps, so they are found in the ledger only if block 1's
counts are the recorded ones; that they are found is itself the check that the pass is the one that ran. Each command
prints the answer and the evidence (exact against device) and writes
`results/<TAG>_device_ibm_kingston_2000_i<pair>...json`; the numbers equal those of the paper and of the bootstraps
above (pair 3765: entailment 0.925 against 0.905 exact, evidence correlation 0.997).

## 2. Repeat the jobs with your own account

### The keys file

The scripts take your IBM Quantum API key from a small text file that you keep outside the repository and never
commit, one line `<name> <API key>` (the name is only a label, an e-mail address or anything without spaces). The
examples below call it `~/ibm_keys.txt`.

```bash
python src/hardware/ibm_jobs.py check --keys ~/ibm_keys.txt     # logs in (sends no job): usable or not, quota left
```

`check` writes `ibmq_accounts_status.json` next to the keys file, never into the repository, and keeps no key in it.
A new IBM account needs an instance first: on IBM Quantum Platform, create one on the free (Open) plan, which has 10
minutes of quantum time in any 28 days. On ibm_kingston one of these circuits at 2,000 shots used 0.54 quantum
seconds in our jobs, so each pair (2 x 288 circuits) took 310 s and the deletion (2 x 210) 228 s.

### Why running the same command again sends nothing

The ledger names each job by a key over its circuits, the backend and the shots. If that key is already in
`ibm_jobs/` with its counts, the scripts take the counts from there and send nothing, so that no job is ever paid for
twice by mistake; a new job whose circuits were already sent in another job is refused. To measure the same circuits
again on purpose, give a repeat number: `--repeat 1` (ours are 0). A repeat is a new key and a new job.

### Dry run, then send

Without `--submit` the command is a dry run: it logs in, checks your quota, transpiles the circuits for the
backend's current calibration and writes the job folder in `ibm_jobs/` (status "prepared"), but sends nothing; it
goes on with ideal counts so that block 2 can be built, and says `dry run (ideal counts)`. Then the same command
with `--submit` sends it. Pair 3765 again, about 5 minutes of quantum time:

```bash
python src/hardware/hw_attention.py $TAG --index 3765 --shots 2000 --device ibm_kingston --cal-random 32 --keys ~/ibm_keys.txt --repeat 1
python src/hardware/hw_attention.py $TAG --index 3765 --shots 2000 --device ibm_kingston --cal-random 32 --keys ~/ibm_keys.txt --repeat 1 --submit
```

The two blocks are two jobs, one after the other: block 2's circuits are built from block 1's counts, so a repeat's
block 2 differs from ours. The script stops before sending when the account has too little quota left. A job that was
sent but whose results were not collected (the script was closed while it waited) is fetched, not sent again, the
next time the same command runs. Without `--cal-random` the run uses self-overlap calibrations, which the transpiler
makes shallow and which miss most of the loss; `--cal-random 32` adds 32 calibration circuits per block as deep as
the overlaps (the deletion run asks only for the answer and does not use its calibrations).
`--index` takes any test pair; `src/hardware/hw_plan.py` says which pairs the evidence is likely to survive.

### Another machine

`--device <name>` sends to another backend; a different backend is a different key, so the circuits are transpiled
anew for it and nothing of ours is reused.

### What to expect

A repeat will not give our numbers to the digit. The chip's errors change from day to day: compare the
`properties.json` in your job folder (qubit T1/T2, gate and readout errors when it was sent) with ours, and the
qubits each circuit ran on (`record.json`). `python src/hardware/device_overlaps.py <your job key>` shows its visibility and
offset; ours moved between 0.66 and 0.82 from job to job. To add your job to a copy of this repository, commit its
`ibm_jobs/<key>/` folder; `python src/hardware/ibm_jobs.py list` rewrites `ibm_jobs/index.md`. Your folder records a fingerprint
of the key that sent the job (the first 12 hex digits of its SHA-256), never the key itself.

## 3. Plain Qiskit, without our scripts

Each job folder holds the circuits as built (`circuits.qpy`) and as sent to ibm_kingston (`transpiled.qpy`). The
sketch below runs a recorded job again with IBM's own Sampler; it uses the transpiled circuits, so it works only on
ibm_kingston (for another machine transpile `circuits.qpy` for it first):

```python
import json
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

job_dir = "ibm_jobs/a1d6be83e81a3267"                    # any folder of ibm_jobs/; this one is block 1 of pair 3765
with open(f"{job_dir}/transpiled.qpy", "rb") as f:
    circuits = qpy.load(f)
ours = json.load(open(f"{job_dir}/counts.json"))         # our counts, one dictionary per circuit

service = QiskitRuntimeService(channel="ibm_quantum_platform", token="<your API key>")
job = SamplerV2(mode=service.backend("ibm_kingston")).run(circuits, shots=2000)
yours = [r.join_data().get_counts() for r in job.result()]
```

`yours[i]` and `ours[i]` are then the counts of the same circuit; each circuit measures one qubit (the Hadamard
test's ancilla), and `(counts["0"] - counts["1"]) / 2000` is its reading, the real or imaginary part of an overlap
times the visibility (which circuit reads what: the `note` of `record.json`'s `circuits`). To try the sketch without
an account, `from qiskit_ibm_runtime.fake_provider import FakeKingston` and `SamplerV2(mode=FakeKingston())` run it on
a local simulation of the device's noise (milder than the device's: readings come out larger). This is written for
qiskit-ibm-runtime 0.51, which the paper used; it warns that `SamplerV2` will be renamed in a later release.
