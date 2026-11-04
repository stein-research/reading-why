# Checking and repeating the IBM jobs

Every hardware number in the paper comes from jobs run on IBM's ibm_kingston, and every job is recorded in
[`ibm_jobs/`](../ibm_jobs/): the circuits as built and as sent, the backend's calibration at the time and the raw
counts (what each file holds: [`ibm_jobs/README.md`](../ibm_jobs/README.md); the list of jobs:
[`ibm_jobs/index.md`](../ibm_jobs/index.md)). This page shows three things, from the least to the most work:

1. check the paper's hardware numbers from those records, with no IBM account;
2. run the same circuits again on IBM hardware with your own account;
3. send the recorded circuits with plain Qiskit, without our scripts.

Install first as in [`run-locally.md`](run-locally.md) (the full `requirements.txt`). Every command runs from the root
of the clone. `TAG` is the model the hardware jobs used (the Born Transformer, seed 0, 20 epochs):

```bash
TAG=esnli_Born-p2-ap_b2_s0_e20
```

The scripts that rebuild a pass need e-SNLI's test split, which `src/esnli.py` downloads into `data/esnli/` the first
time (see [`data/README.md`](../data/README.md)).

## 1. Check the numbers, no account

These read only the files in the repository and send nothing.

| in the paper | command | prints | time |
|---|---|---|---|
| quantum seconds of every job (Table 11) | `python src/ibm_jobs.py list` | the job list, rewritten into `ibm_jobs/index.md` | seconds |
| the overlaps against their ideal values (6): correlation 0.992 over the 256 of block 1 of pair 3763, visibility 0.82 in the pilot and 0.81 as that slope; 0.82 / 0.81 fitted to the deep calibrations of pair 4667 | `python src/device_overlaps.py` (or name job keys: `... 9333cd8540d5532d cf7fcc3a3c6371ca`) | per job and kind of circuit: correlation, slope (the visibility), offset | a minute |
| the device runs of Table 5 (answers of pairs 3763 and 3765, with and without a word) and their 95 % intervals | `python src/device_bootstrap.py` (`--only "pair 3763"` for one run) | each run's P on the device, its interval, and the share of redraws that keep the answer | about 15 min a run |
| the evidence on the device, pair 4667 (6, Table 6, Fig. 1b): correlation 0.997, interval 0.987-0.999, the same top word in every redraw, every sign in 65 %, slopes 0.82 / 0.81 and offsets +0.004 / +0.012; pair 997 (6): 0.932, interval 0.73-0.99, every sign in every redraw | `python src/evidence_bootstrap.py` (pair 4667; `--index 997` for the other; 500 redraws, `--draws 20` to see the device values in a minute) | the device's evidence, each word's interval, the correlation and its interval, the fitted calibrations | about an hour |
| the earlier runs replayed under other calibrations (6): pair 3765 0.95 every sign, 3763 0.81, with one visibility per block | `python src/hw_recal.py $TAG --index 3765` (and `--index 3763`) | the evidence under each calibration, with its correlation | a few minutes |
| the choice of pair 4667 in advance (6): the ten shortest test pairs under a simulation of the device | `python src/hw_plan.py $TAG --max-tokens 9 --draws 12` | per pair: cancellation, median and 10th-percentile correlation, signs, circuits, quantum seconds | hours on a CPU |

The bootstraps and the replay read the counts in `ibm_jobs/` directly. To rebuild a whole pass from the raw counts
with the script that sent it, run its device command without `--submit`: it rebuilds every circuit from the model,
finds each job in `ibm_jobs/` by its key, takes the counts from there, and logs in nowhere and sends nothing (it
prints `already done ... counts from the ledger, nothing sent` for each block):

```bash
python src/hw_attention.py $TAG --index 4667 --shots 2000 --device ibm_kingston --cal-random 32
python src/hw_attention.py $TAG --index 997 --shots 2000 --device ibm_kingston --cal-random 32
python src/hw_attention.py $TAG --index 3763 --shots 2000 --device ibm_kingston
python src/hw_attention.py $TAG --index 3763 --shots 2000 --device ibm_kingston --delete 5 --no-evidence
```

Block 2's circuits are built from block 1's measured overlaps, so they are found in the ledger only if block 1's
counts are the recorded ones; that they are found is itself the check that the pass is the one that ran. Each command
prints the answer and the evidence (exact against device) and writes
`results/<TAG>_device_ibm_kingston_2000_i<pair>...json`; the numbers equal those of the paper and of the bootstraps
above (pair 4667: contradiction 0.980 against 0.969 exact, evidence correlation 0.997).

## 2. Repeat the jobs with your own account

### The keys file

The scripts take your IBM Quantum API key from a small text file that you keep outside the repository and never
commit: one account per line, `<name> <API key>` (the name is only a label, an e-mail address or anything without
spaces). The examples below call it `~/ibm_keys.txt` and use its first line, `--account 0`.

```bash
python src/ibm_jobs.py check --keys ~/ibm_keys.txt     # logs in (sends no job): usable or not, quota left
```

`check` writes `ibmq_accounts_status.json` next to the keys file, never into the repository, and keeps no key in it.
A new IBM account needs an instance first: on IBM Quantum Platform, create one on the free (Open) plan, which has 10
minutes of quantum time in any 28 days. On ibm_kingston one of these circuits at 2,000 shots used 0.54 quantum
seconds in our jobs, so pair 4667 (864 circuits) took 464 s and pair 3763 (544) 294 s.

### Why running the same command again sends nothing

The ledger names each job by a key over its circuits, the backend and the shots. If that key is already in
`ibm_jobs/` with its counts, the scripts take the counts from there and send nothing, so that no job is ever paid for
twice by mistake; a new job whose circuits were already sent in another job is refused. To measure the same circuits
again on purpose, give a repeat number: `--repeat 1` (ours are 0). A repeat is a new key and a new job.

### Dry run, then send

Without `--submit` the command is a dry run: it logs in, checks your quota, transpiles the circuits for the
backend's current calibration and writes the job folder in `ibm_jobs/` (status "prepared"), but sends nothing; it
goes on with ideal counts so that block 2 can be built, and says `dry run (ideal counts)`. Then the same command
with `--submit` sends it. Pair 4667 again, about 8 minutes of quantum time:

```bash
python src/hw_attention.py $TAG --index 4667 --shots 2000 --device ibm_kingston --cal-random 32 --keys ~/ibm_keys.txt --account 0 --repeat 1
python src/hw_attention.py $TAG --index 4667 --shots 2000 --device ibm_kingston --cal-random 32 --keys ~/ibm_keys.txt --account 0 --repeat 1 --submit
```

The two blocks are two jobs, one after the other: block 2's circuits are built from block 1's counts, so a repeat's
block 2 differs from ours. The script stops before sending when the account has too little quota left. A job that was
sent but whose results were not collected (the script was closed while it waited) is fetched, not sent again, the
next time the same command runs. Without `--cal-random` the run uses the self-overlap calibrations of the first runs,
which the transpiler makes shallow; `--cal-random 32` adds 32 calibration circuits per block as deep as the overlaps.
`--index` takes any test pair; `src/hw_plan.py` says which pairs the evidence is likely to survive.

### Another machine

`--device <name>` sends to another backend; a different backend is a different key, so the circuits are transpiled
anew for it and nothing of ours is reused.

### What to expect

A repeat will not give our numbers to the digit. The chip's errors change from day to day: compare the
`properties.json` in your job folder (qubit T1/T2, gate and readout errors when it was sent) with ours, and the
qubits each circuit ran on (`record.json`). `python src/device_overlaps.py <your job key>` shows its visibility and
offset; ours moved between 0.75 and 0.83 from job to job. To add your job to a copy of this repository, commit its
`ibm_jobs/<key>/` folder; `python src/ibm_jobs.py list` rewrites `ibm_jobs/index.md`. Your folder records which line
of your keys file sent the job, and a fingerprint of the key (the first 12 hex digits of its SHA-256), never the key
itself.

## 3. Plain Qiskit, without our scripts

Each job folder holds the circuits as built (`circuits.qpy`) and as sent to ibm_kingston (`transpiled.qpy`). The
sketch below runs a recorded job again with IBM's own Sampler; it uses the transpiled circuits, so it works only on
ibm_kingston (for another machine transpile `circuits.qpy` for it first):

```python
import json
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

job_dir = "ibm_jobs/f5ddb65f08d376c5"                    # any folder of ibm_jobs/; this one is block 1 of pair 4667
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
