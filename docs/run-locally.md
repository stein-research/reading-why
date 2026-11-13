# Running Reading Why on your own computer

The notebooks run in Colab with no install (links in the [README](../README.md)). This page is for running them, and
the paper's scripts, from a clone.

## What you need

- Python 3.10 or newer (the paper used 3.12) and git.
- About 250 MB of disk for the clone (the trained models and the IBM job records included) and 200 MB for e-SNLI. The
  packages take about 6 GB with PyTorch's default build on Linux, which brings its CUDA libraries; with the CPU-only
  build (below) they take much less.
- A CPU is enough for the demo and for the lighter scripts below. A GPU (CUDA) makes training and the evaluations
  minutes instead of hours.

## Install

```bash
git clone https://github.com/stein-research/reading-why
cd reading-why
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyterlab
```

`requirements.txt` lists what the demo needs first, then what the circuits, the IBM scripts and the optics check add.
For the demo alone, the first block and `jupyterlab` are enough. Without a GPU, install PyTorch's CPU-only build first,
which is much smaller, and then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt jupyterlab
```

## The notebooks

```bash
jupyter lab demo/demo.ipynb        # step through the test pairs or type your own; evidence, attention, deletion
jupyter lab demo/train.ipynb       # train one of the paper's models (a GPU helps)
```

Run the cells in order. The setup cell finds the clone. The demo downloads only the e-SNLI test split (7 MB, checked
by its SHA-256) into `data/esnli/`; the training notebook downloads all of e-SNLI (about 200 MB, each file checked).
VS Code's notebook editor works too (it needs `ipywidgets`, which `requirements.txt` installs). If the controls do not
show, restart the kernel and run the setup cell again.

## In plain Python

The code below is [`examples/quickstart.py`](../examples/quickstart.py); run it with `python examples/quickstart.py`
from the repository's top folder:

```python
import sys; sys.path.insert(0, "demo")
import reading_why as rw

m = rw.load("born")                    # the Born Transformer; or "softmax", or the tag of a run in results/
r = rw.explain(m, "A woman working long hours.", "A woman is sleeping.")
print(r["answer"], r["probs"].round(3))   # contradiction [0.004 0.001 0.995]
for w, e, a in zip(r["words"], r["evidence"], r["attention"]):
    print(f"{w:>10s}  evidence {e:+.4f}  attention {a:.3f}")
print(r["evidence"].sum(), r["explained"])            # equal (0.662817): the evidence sums to P_g - b_g

ids, words = rw.without(r, [r["words"].index("sleeping")])   # delete words in place, as the paper does
print(rw.explain(m, ids=ids, toks=words)["answer"])          # entailment
```

[`examples/explain.py`](../examples/explain.py) does the same for any pair from the command line:

```bash
python examples/explain.py "A woman working long hours." "A woman is sleeping." --delete sleeping
python examples/explain.py --test 3763 --model softmax          # an e-SNLI test pair, with the annotators' words
```

## Train a model

Every script downloads the e-SNLI files it needs the first time (from the authors' repository at the paper's commit,
about 200 MB in all, each file checked against `data/SHA256SUMS`; a file that does not match is refused). This is the
command the training notebook runs, here for the paper's Born Transformer (20 epochs: about an hour on a laptop GPU):

```bash
python src/train.py --model Born-ap --blocks 2 --epochs 20 --lr 2e-3 --drop 0.1 --seed 3 --data esnli \
  --tag esnli_Born-ap_b2_s3_e20
```

Other models: `Born-p2-ap` (the attention weight squared again, the second row of Table 1), `Softmax`,
`Poly-sim-born`, the ablations `Poly-softmax-bil`, `Poly-born-gelu`, `Poly-softmax-gelu`,
and the coherent attention layer `Coherent-attn-b12-24` (`--blocks 1 --epochs 10`). The run is written to `results/`
(`<tag>.json` and `<tag>.pt`); `rw.load("<tag>")` opens it. Seeds 0-2 are the paper's runs, so pick 3 or more for
your own.

## The paper's analyses

Each script prints its numbers and writes them to `results/`. The trained models of the tables are already there
(the Born Transformer, the softmax transformer and Poly-sim-born, three seeds each; the attention weight squared again
in `results/ablation/`; see [`results/README.md`](../results/README.md)), so nothing needs training first except the
ablations of Table 3. `TAGS` below stands for the nine main runs (add
`ablation/esnli_Born-p2-ap_b2_s{0,1,2}_e20` for the second row of Table 1):

```bash
TAGS=$(for m in Born-ap Softmax Poly-sim-born; do for s in 0 1 2; do echo esnli_${m}_b2_s${s}_e20; done; done)
```

| script | what it gives (paper section) | rough time |
|---|---|---|
| `python src/faithfulness.py $TAGS` | gap, deletion (AOPC, flip) and agreement (Table 1) | an hour on a CPU, minutes on a GPU |
| `python src/fairness.py` then `python src/gradinput.py` | every ranking, the leave-one-out ceilings, 95 % intervals (Table 2) | hours on a CPU, about 30 min on a GPU |
| `python src/attention_agreement.py $TAGS` | attention against the annotators and the answer, rollout (Table 2) | minutes |
| `python src/ig_nodes.py` | the softmax model's integrated gradients of its logit over the embeddings with 256 to 4,096 nodes, seed 0, first 200 pairs (2.2) | long; use a GPU |
| `python src/gated_converged.py` (after `fairness.py`) then `python src/two_level.py` and `python src/gated_baseline.py` | the like-for-like comparison's word-gated probability IG at 4,096 nodes on all 1,000 pairs and three seeds, flip@1 on the converged pairs, the two-level (seeds and pairs) intervals, and the softmax model's answer with every gate at zero (3.3, Table 2); `results/gated_baseline.json` holds our run of the last | long; use a GPU |
| train the three ablations (3 seeds each), then `faithfulness.py` and `attention_agreement.py` on them | which part gives the evidence its quality (Table 3) | 30-50 min per run on a GPU |
| `python src/compile_hw.py esnli_Born-ap_b2_s0_e20` (add `--noise` for shot noise and bits) | the compiled optics (4) | minutes |
| `python src/mesh_check.py esnli_Born-ap_b2_s0_e20 --n 1000` | the build with Clements meshes, with phase errors (4); needs `interferometer` | about an hour (the 256-mode decomposition is cached) |
| `python src/coherent_circuit.py esnli_Coherent-attn-b12-24_b1_s0 --index 3765 3763 --gpu` | the coherent layer as one circuit, equal to the model (Table 9, Fig. 8); `--cost-only 3` counts gates only | needs qiskit-aer-gpu (`QML_GPU_PYTHON`) for the 23-qubit statevector |
| `python src/hw_attention.py esnli_Born-ap_b2_s0_e20 --n 50` (and `--evidence 20`) | the answer and the hybrid evidence under FakeKingston, the noise model of ibm_kingston (5) | hours |
| `python src/qubit_resources.py --n 300` (`--tag ablation/esnli_Born-p2-ap_b2_s0_e20` for the weight squared again) | shots, visibility, calibration and the coherent layer's post-selection (5 and 6, Tables 8 and 9) | about an hour on a GPU |
| `python src/device_bootstrap.py` and `python src/evidence_bootstrap.py --index 3763` (and `3765`) | shot-noise intervals of every device run's answer and of the evidence, from the recorded counts (5, Tables 4 and 6, Fig. 7) | 15 min per run on a CPU; the evidence about an hour |
| `python src/deletion_words.py` | every single-word deletion of the two device pairs in the three seeds, with each pair's evidence (5) | minutes |
| `python src/nodes_check.py` | the evidence with 30, 100 and 400 quadrature nodes on ten pairs: 100 nodes are a bound, not a fine tuning (2.2) | minutes |
| `python src/bow_baseline.py --data esnli` | the bag-of-words references (3.2) | minutes |
| `python src/bag_baseline.py inter <seed> 20`, then `python src/bag_evidence.py` | the order-free bag of embeddings (Table 1), and its own evidence: gradient x input, integrated gradients, leave-one-out (3.3) | about 7 min per seed on a GPU |
| `python src/esnli_baselines.py` and `python src/esnli_seed_stability.py` | rankings that need no model (word position, rare words), where the evidence's first token lies, the model's scale dependence; the evidence across training seeds (3.3, 5) | minutes on a GPU |
| `python src/photon_draws.py` | the optics under shot noise over 20 draws, and the photons at each detector (4) | minutes |
| `python src/device_power.py`, `src/device_blocks.py`, `src/device_rows.py`, `src/device_errors.py`, `src/device_floor.py`, `src/device_rowsd.py`, `src/device_block2_nulls.py` | what the device checks can show: uniform and random attention, block by block; which row sums are held; the error per circuit against shot noise; nulls for the evidence check (5) | minutes each on a CPU |
| `python src/cu_analysis.py` | the compute-uncompute runs: answers, calibrations, evidence and their shot-noise intervals (5) | about 15 min on a CPU |
| `python src/hw_cu_compare.py` | Hadamard tests against compute-uncompute on the same 128 Born weights of pair 3765's first block (5, Table 7) | a minute on a CPU |
| `python src/shot_bound.py` then `python src/device_shot_floor.py` | the proposition on the shots the evidence needs, checked on 100 pairs, and against the device (5, Appendix C); `results/shot_bound.json` holds our run | 30 min on a GPU |
| `python src/coherent23.py esnli_Coherent-attn-b12-24_b1_s0` | the coherent circuit as counted: 23 qubits, B and C apart, every state prepared by gates (6) | about 5 min per pair on a CPU |
| `python src/softmax_held.py` (after `gated_converged.py` and `fairness.py`) | the softmax model's integrated gradients with its denominators held: on how many pairs they are finite and complete, and every ranking on those pairs (3.3); `results/softmax_held.json` holds our run | about an hour on a GPU |
| `python src/path_check.py` | the held path of each pair: how steeply it rises near its end (k of alpha^k), its peak, and gradient x input with the denominators held against BEW (3.3); `results/path_check.json` holds our run | about an hour on a GPU |
| `python src/free_denominators.py --nodes 1024 --max-tok 14` then `python src/fig_free_path.py` | the evidence against integrated gradients with the denominators free and the exact Shapley value of the deletion game on pairs of at most 14 tokens; the free path along alpha (Appendix B, Fig. 9); `results/free_denominators_n1024_p1000.json` holds our run | about 1.5 h on a GPU, much longer on a CPU |

## The IBM scripts

Checking every hardware number from the recorded jobs (no account), repeating the jobs with your own account, and
sending the recorded circuits with plain Qiskit: [check-and-repeat-ibm.md](check-and-repeat-ibm.md).

`src/hw_attention.py --device` sent the paper's jobs through the ledger `src/ibm_jobs.py`, which never sends a job
twice and checks the quota first. Running them again needs your own IBM Quantum account, passed as a keys file
(`--keys`, one line `<name> <API key>`, kept outside the repository); no account details are in this repository.
Without `--submit` it makes a dry run (transpiled and recorded, nothing sent); `--repeat 1` makes it a new job rather
than the recorded one, which the ledger would otherwise read back:

```bash
python src/hw_attention.py esnli_Born-ap_b2_s0_e20 --index 3763 --shots 2000 --device ibm_kingston --cal-random 32 --keys <file> --repeat 1
```

Every job the paper used is recorded in [`ibm_jobs/`](../ibm_jobs/), with its circuits, transpiled circuits,
calibration and raw counts, so the hardware numbers can be checked without an account: `python src/ibm_jobs.py list`,
`python src/ibm_jobs.py show <key or job id>`, and `src/device_bootstrap.py` read those records.

## Files in `src/` and `demo/`

| file | what |
|---|---|
| `src/models.py` | every model: the Born Transformer, the baselines, the ablations, the coherent attention layers (brickwall maps) |
| `src/evidence.py`, `src/nodes_check.py` | the exact evidence: integrated gradients with the denominators held, Gauss-Legendre quadrature; the check that fewer nodes already agree |
| `src/train.py` | training (one recipe for every model) |
| `src/esnli.py`, `src/data.py`, `src/synth.py` | e-SNLI tokens, highlights and vocabulary; the tokenizer; dataset access |
| `src/faithfulness.py`, `src/fairness.py`, `src/gradinput.py`, `src/attention_agreement.py`, `src/ig_nodes.py`, `src/gated_converged.py`, `src/gated_baseline.py`, `src/path_check.py`, `src/softmax_held.py` | the evaluations of the evidence |
| `src/compile_hw.py`, `src/mesh_check.py` | the compile to optics and qubits; the Clements-mesh check |
| `src/hw_attention.py`, `src/ibm_jobs.py`, `src/device_bootstrap.py`, `src/evidence_bootstrap.py`, `src/device_overlaps.py`, `src/hw_plan.py`, `src/deletion_words.py` | Hadamard tests on a noise model or an IBM device; the job ledger; shot-noise intervals of the answers and of the evidence; the recorded overlaps against their ideal values; a simulation of the device with the noise the recorded runs show; every single-word deletion of the device pairs |
| `src/coherent_circuit.py`, `src/sim_gpu.py`, `src/qubit_resources.py` | the coherent layer as one circuit; GPU statevector; quantum resources and errors |
| `src/bow_baseline.py`, `src/bag_baseline.py`, `src/bag_evidence.py`, `src/esnli_baselines.py`, `src/esnli_seed_stability.py` | the bag-of-words references; the bag of embeddings and its evidence; model-free rankings; stability across seeds |
| `src/device_power.py`, `src/device_blocks.py`, `src/device_rows.py`, `src/device_errors.py`, `src/device_floor.py`, `src/device_rowsd.py`, `src/device_block2_nulls.py`, `src/cu_analysis.py`, `src/hw_cu_compare.py`, `src/shot_bound.py`, `src/device_shot_floor.py`, `src/photon_draws.py`, `src/coherent23.py` | what the device checks show; the compute-uncompute runs; the shots proposition; the optics' photons; the 23-qubit coherent circuit |
| `src/free_denominators.py`, `src/fig_free_path.py` | the evidence against free-denominator attributions and the deletion game's Shapley value (Appendix B) |
| `examples/quickstart.py`, `examples/explain.py` | the demo in plain Python; one pair from the command line |
| `demo/reading_why.py` | the demo's engine: load a model, explain a pair (evidence by quadrature, nodes run as one batch), attention, deletion, the e-SNLI download |
| `demo/ui.py` | the widgets and the HTML card |
