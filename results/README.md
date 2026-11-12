# Trained models

Model names: `Born-ap` is the Born Transformer, attention weight |<q|k>|^2 over its row sum; `Born-p2-ap` squares
that weight again ("p2": the Born probability to the power 2, so |<q|k>|^4); "ap" stands for the additive position
fields both Born models use.

Each run is a pair: `<tag>.pt` (the weights, PyTorch `state_dict`) and `<tag>.json` (its settings, training
history and test accuracy). Tags read `esnli_<model>_b<blocks>_s<seed>_e<epochs>`. All were trained on e-SNLI with
`src/train.py` (20 epochs, Adam 2e-3, cosine decay, batch 64, word dropout 0.1).

| model | runs | test accuracy (3 seeds) | in the paper |
|---|---|---|---|
| `Born-ap` (the Born Transformer: attention weight = squared overlap over its row sum) | seeds 0-2 | 77.45 +- 0.91 % | everything; seed 0 on the device and in the figures |
| `ablation/Born-p2-ap` (the same with the weight squared again, fourth power of the overlap; the main model of arXiv v1-v2) | seeds 0-2 | 76.98 +- 0.17 % | the second row of Table 1 |
| `Softmax` (the softmax transformer) | seeds 0-2 | 76.62 +- 0.36 % | the baseline (Tables 1, 2) |
| `Poly-sim-born` (polynomial, no measurement constraints) | seeds 0-2 | 78.98 +- 0.31 % | Tables 1, 2, 3 |
| `Coherent-attn-b12-24` (the coherent attention layer, brickwall maps) | seed 0, 10 epochs | 67.2 % (seeds 0-2: 67.04 +- 0.12 %) | Section 6; the circuit; qubit resources |
| `bag_inter` (an order-free bag of embeddings, no attention; `src/bag_baseline.py`) | seeds 0-2 (weights: seed 0) | 77.79 +- 0.26 % | the baseline row of Table 1; its own evidence (`src/bag_evidence.py`) |

The ablation's runs are in `ablation/`; scripts take their tag with the folder, e.g. `ablation/esnli_Born-p2-ap_b2_s0_e20`.

`esnli_vocab.json` is the vocabulary (22,319 entries; CC BY-SA 4.0, see `../data/README.md`).

The ablations of Table 3 (`Poly-softmax-bil`, `Poly-born-gelu`, `Poly-softmax-gelu`) are not stored; train them
with `python src/train.py --model <name> --blocks 2 --epochs 20 --data esnli --seed <s>` (about 30-50 min each on a
laptop GPU). Scripts in `../src/` write their outputs (`*_faith.json`, `fairness.json`, ...) next to these files; `shot_bound.json`
(`src/shot_bound.py`, 30 min on a GPU) is stored.
