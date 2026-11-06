# Running Reading Why on your own computer

The notebooks run in Colab with no install (links in the [README](../README.md)). This page is for running them, and
the paper's scripts, from a clone.

## What you need

- Python 3.10 or newer (the paper used 3.12) and git.
- About 100 MB of disk for the clone (the trained models included), 200 MB for e-SNLI, and a few hundred MB for the
  packages.
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
For the demo alone, the first block and `jupyterlab` are enough.

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
print(r["answer"], r["probs"])         # contradiction [0.019 0.006 0.975]
for w, e, a in zip(r["words"], r["evidence"], r["attention"]):
    print(f"{w:>10s}  evidence {e:+.4f}  attention {a:.3f}")
print(r["evidence"].sum(), r["explained"])            # equal (0.643164): the evidence sums to P_g - b_g

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
python src/train.py --model Born-p2-ap --blocks 2 --epochs 20 --lr 2e-3 --drop 0.1 --seed 3 --data esnli \
  --tag esnli_Born-p2-ap_b2_s3_e20
```

Other models: `Softmax`, `Poly-sim-born`, the ablations `Poly-softmax-bil`, `Poly-born-gelu`, `Poly-softmax-gelu`,
and the coherent attention layer `Coherent-attn-b12-24` (`--blocks 1 --epochs 10`). The run is written to `results/`
(`<tag>.json` and `<tag>.pt`); `rw.load("<tag>")` opens it. Seeds 0-2 are the paper's runs, so pick 3 or more for
your own.

## The paper's analyses

Each script prints its numbers and writes them to `results/`. The nine trained models of the tables are already there
(three models, three seeds; see [`results/README.md`](../results/README.md)), so nothing needs training first except
the ablations. `TAGS` below stands for those nine:

```bash
TAGS=$(for m in Born-p2-ap Softmax Poly-sim-born; do for s in 0 1 2; do echo esnli_${m}_b2_s${s}_e20; done; done)
```

| script | what it gives (paper section) | rough time |
|---|---|---|
| `python src/faithfulness.py $TAGS` | gap, deletion (AOPC, flip) and agreement (Table 1) | an hour on a CPU, minutes on a GPU |
| `python src/fairness.py` then `python src/gradinput.py` | every ranking, the leave-one-out ceilings, 95 % intervals (Table 2) | hours on a CPU, about 30 min on a GPU |
| `python src/attention_agreement.py $TAGS` | attention against the annotators and the answer, rollout (Table 2) | minutes |
| `python src/ig_nodes.py` | the softmax model's integrated gradients of its logit over the embeddings with 256 to 4,096 nodes, seed 0, first 200 pairs (2.2) | long; use a GPU |
| `python src/gated_converged.py` (after `fairness.py`) then `python src/two_level.py` and `python src/gated_baseline.py` | the like-for-like comparison's word-gated probability IG at 4,096 nodes on all 1,000 pairs and three seeds, flip@1 on the converged pairs, the two-level (seeds and pairs) intervals, and the softmax model's answer with every gate at zero (3.3, Table 2) | long; use a GPU |
| train the three ablations (3 seeds each), then `faithfulness.py` and `attention_agreement.py` on them | which part gives the evidence its quality (Table 3) | 30-50 min per run on a GPU |
| `python src/compile_hw.py esnli_Born-p2-ap_b2_s0_e20` (add `--noise` for shot noise and bits) | the compiled optics (4) | minutes |
| `python src/mesh_check.py esnli_Born-p2-ap_b2_s0_e20 --n 1000` | the build with Clements meshes, with phase errors (4); needs `interferometer` | about an hour (the 256-mode decomposition is cached) |
| `python src/coherent_circuit.py esnli_Coherent-attn-b12-24_b1_s0 --index 3765 3763 --gpu` | the coherent layer as one circuit, equal to the model (Table 4, Fig. 6); `--cost-only 3` counts gates only | needs qiskit-aer-gpu (`QML_GPU_PYTHON`) for the 23-qubit statevector |
| `python src/hw_attention.py esnli_Born-p2-ap_b2_s0_e20 --n 50` (and `--evidence 20`) | the answer and the hybrid evidence under the noise model of ibm_torino (6) | hours |
| `python src/qubit_resources.py --n 300` | shots, visibility, calibration and the coherent layer's post-selection (6, Tables 7 and 8) | about an hour on a GPU |
| `python src/device_bootstrap.py` | shot-noise intervals of every device run, from the recorded counts (6, Fig. 8) | 15 min per run on a CPU |
| `python src/bow_baseline.py --data esnli` | the bag-of-words references (3.2) | minutes |
| `python src/free_denominators.py --nodes 1024 --max-tok 14` then `python src/fig_free_path.py` | the evidence against integrated gradients with the denominators free and the exact Shapley value of the deletion game on pairs of at most 14 tokens; the free path along alpha (Appendix B, Fig. 9); `results/free_denominators_n1024_p1000.json` holds our run | about 1.5 h on a GPU, much longer on a CPU |

## The IBM scripts

Checking every hardware number from the recorded jobs (no account), repeating the jobs with your own account, and
sending the recorded circuits with plain Qiskit: [check-and-repeat-ibm.md](check-and-repeat-ibm.md).

`src/hw_attention.py --device` sent the paper's jobs through the ledger `src/ibm_jobs.py`, which never sends a job
twice and checks the quota first. Running them again needs your own IBM Quantum account, passed as a keys file
(`--keys`, one `<email> <token>` per line, kept outside the repository); no account details are in this repository.
Without `--submit` it makes a dry run (transpiled and recorded, nothing sent):

```bash
python src/hw_attention.py esnli_Born-p2-ap_b2_s0_e20 --index 3763 --shots 2000 --device ibm_kingston --keys <file> --account 0
```

Every job the paper used is recorded in [`ibm_jobs/`](../ibm_jobs/), with its circuits, transpiled circuits,
calibration and raw counts, so the hardware numbers can be checked without an account: `python src/ibm_jobs.py list`,
`python src/ibm_jobs.py show <key or job id>`, and `src/device_bootstrap.py` read those records.

## Files in `src/` and `demo/`

| file | what |
|---|---|
| `src/models.py` | every model: the Born Transformer, the baselines, the ablations, the coherent attention layers (brickwall maps) |
| `src/evidence.py` | the exact evidence: integrated gradients with the denominators held, Gauss-Legendre quadrature |
| `src/train.py` | training (one recipe for every model) |
| `src/esnli.py`, `src/data.py`, `src/synth.py` | e-SNLI tokens, highlights and vocabulary; the tokenizer; dataset access |
| `src/faithfulness.py`, `src/fairness.py`, `src/gradinput.py`, `src/attention_agreement.py`, `src/ig_nodes.py`, `src/gated_converged.py`, `src/gated_baseline.py` | the evaluations of the evidence |
| `src/compile_hw.py`, `src/mesh_check.py` | the compile to optics and qubits; the Clements-mesh check |
| `src/hw_attention.py`, `src/ibm_jobs.py`, `src/device_bootstrap.py`, `src/evidence_bootstrap.py`, `src/device_overlaps.py`, `src/hw_recal.py`, `src/hw_plan.py` | Hadamard tests on a noise model or an IBM device; the job ledger; shot-noise intervals of the answers and of the evidence; the recorded overlaps against their ideal values; the earlier runs replayed under other calibrations; the device simulation that chose pair 4667 |
| `src/coherent_circuit.py`, `src/sim_gpu.py`, `src/qubit_resources.py` | the coherent layer as one circuit; GPU statevector; quantum resources and errors |
| `src/bow_baseline.py` | the bag-of-words references |
| `src/free_denominators.py`, `src/fig_free_path.py` | the evidence against free-denominator attributions and the deletion game's Shapley value (Appendix B) |
| `examples/quickstart.py`, `examples/explain.py` | the demo in plain Python; one pair from the command line |
| `demo/reading_why.py` | the demo's engine: load a model, explain a pair (evidence by quadrature, nodes run as one batch), attention, deletion, the e-SNLI download |
| `demo/ui.py` | the widgets and the HTML card |
