# The code, and how to reproduce the paper

Run every command from the repository root; the scripts read and write `results/`, and e-SNLI goes in `data/esnli/`.

## Setup

```bash
git clone https://github.com/stein-research/reading-why
cd reading-why
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # the first block is enough for the models and tables
python -c "import sys; sys.path.insert(0, 'demo'); import reading_why as rw; rw.get_esnli()"   # e-SNLI, ~200 MB, checked
```

A GPU (CUDA PyTorch) makes training and the evaluations minutes instead of hours, but every script runs on a CPU.
The nine trained models of the tables are in `results/` (see [`../results/README.md`](../results/README.md)), so the
tables reproduce without training. Tags look like `esnli_Born-p2-ap_b2_s0_e20`; in the commands below `TAGS` stands
for the nine of them:

```bash
TAGS=$(for m in Born-p2-ap Softmax Poly-sim-born; do for s in 0 1 2; do echo esnli_${m}_b2_s${s}_e20; done; done)
```

## Paper item -> command

| paper | command | notes |
|---|---|---|
| Table 1: accuracy | read `test_acc` in `results/<tag>.json` | or retrain, below |
| Table 1: gap, AOPC, flip, agreement | `python src/faithfulness.py $TAGS` | writes `results/<tag>_faith.json`; 1,000 test pairs |
| Sec. 2.2: softmax IG with more nodes | `python src/ig_nodes.py` | 256 / 1,024 / 4,096 nodes, 200 pairs |
| Table 2: every ranking, ceilings, intervals | `python src/fairness.py` then `python src/gradinput.py` | `results/fairness.json`; per-pair `fairness.npz` |
| Table 2: attention against annotators, rollout | `python src/attention_agreement.py $TAGS` | |
| Table 4 and Fig. 6: the coherent layer as a circuit | `python src/coherent_circuit.py esnli_Coherent-attn-b12-24_b1_s0 --index 3765 3763 --gpu` | statevector of 23 qubits; `--gpu` needs qiskit-aer-gpu (set `QML_GPU_PYTHON` to that Python); `--cost-only 3` counts gates only |
| Table 3: which part gives the evidence its quality | train `Poly-softmax-bil`, `Poly-born-gelu`, `Poly-softmax-gelu` (3 seeds each), then `faithfulness.py` and `attention_agreement.py` on them | |
| Sec. 4: the compiled optics; shot noise and bits | `python src/compile_hw.py esnli_Born-p2-ap_b2_s0_e20` and `--noise` | |
| Sec. 4: the outside check with Clements meshes | `python src/mesh_check.py esnli_Born-p2-ap_b2_s0_e20 --n 1000` | needs `interferometer`; the 256-mode decomposition takes ~25 min, cached |
| Sec. 6: the noise model (ibm_torino) | `python src/hw_attention.py esnli_Born-p2-ap_b2_s0_e20 --n 50` ; `--evidence 20` | qiskit-aer, FakeTorino; long |
| Sec. 6 and Fig. 7: the device runs | `python src/ibm_jobs.py list` ; `python src/ibm_jobs.py show <key>` ; `python src/device_bootstrap.py` | from the ledger in `ibm_jobs/`, no IBM account needed |
| Sec. 6 and Table 5: resources and errors | `python src/qubit_resources.py --n 300` | shots, visibility, calibration, the coherent layer's pass rate |

Running on an IBM device yourself needs your own account: a keys file with one `<email> <token>` per line (keep it
outside the repository), then `python src/hw_attention.py <tag> --index 3763 --shots 2000 --device ibm_kingston
--keys <file> --account 0` makes a dry run (transpiled and recorded, nothing sent); add `--submit` to send. The
ledger (`src/ibm_jobs.py`) never sends a job twice and checks the quota first.

## Training

```bash
python src/train.py --model Born-p2-ap --blocks 2 --epochs 20 --lr 2e-3 --drop 0.1 --seed 0 --data esnli --tag esnli_Born-p2-ap_b2_s0_e20
```

Models: `Born-p2-ap` (the Born Transformer), `Softmax`, `Poly-sim-born`, the ablations `Poly-softmax-bil`,
`Poly-born-gelu`, `Poly-softmax-gelu`, and the coherent layer `Coherent-attn-b12-24` (`--blocks 1 --epochs 10`).
About 66 / 18 / 32 min for 20 epochs of the first three on a laptop GPU.

## Files

| file | what |
|---|---|
| `models.py` | every model: the Born Transformer (`BornTransformer`), the baselines, the ablations, the coherent attention layers (`CoherentAttn*`, brickwall maps) |
| `evidence.py` | the exact evidence: integrated gradients with the denominators held, Gauss-Legendre quadrature |
| `train.py` | training (one recipe for every model) |
| `esnli.py`, `data.py`, `synth.py` | e-SNLI tokens, highlights and vocabulary; the tokenizer; dataset access |
| `faithfulness.py`, `fairness.py`, `gradinput.py`, `attention_agreement.py`, `ig_nodes.py` | the evaluations of the evidence |
| `compile_hw.py`, `mesh_check.py` | the compile to optics and qubits; the Clements-mesh check |
| `hw_attention.py`, `ibm_jobs.py`, `device_bootstrap.py` | Hadamard tests on a noise model or an IBM device, the job ledger, shot-noise intervals |
| `coherent_circuit.py`, `sim_gpu.py`, `qubit_resources.py` | the coherent layer as one circuit; GPU statevector; quantum resources and errors |
| `bow_baseline.py` | the bag-of-words references (`--data esnli`) |
