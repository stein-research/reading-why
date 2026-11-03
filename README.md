# Reading Why: Born-Measured Attention with Exact Word Evidence on Light and Qubits

Interactive companion to the paper by Teerawat Tongloy (KMITL), Jiraporn Pongsopa and Rangsan Jomtarak (Suan Dusit
University).

| | |
|---|---|
| Try it: type a premise and a hypothesis and see why the model answers (CPU, no install) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stein-research/reading-why/blob/main/demo/demo.ipynb) |
| Train a model on e-SNLI (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stein-research/reading-why/blob/main/demo/train.ipynb) |
| Paper | arXiv link to follow |
| Earlier work | Seeing Why, [arXiv:2611.01993](https://arxiv.org/abs/2611.01993) ([code](https://github.com/stein-research/seeing-why)) |

![One test pair through the trained model: (a) on light, (b) on IBM's ibm_kingston](docs/overview.png)

*(a) On light (simulated): one field per word on a modulator; in each of two blocks Born attention weighs the words
by interference and a measured feed-forward reads a product by homodyne detection; the camera's three regions give
the class probabilities, and each word's Born evidence (BEW; red for the answer, blue against it) sums to the class
probability above its share. (b) On IBM's ibm_kingston, with the same weights: every query-key overlap is a Hadamard
test on five qubits; the rest is computed classically.*

A transformer in which every step is a Born measurement: attention $|\langle q|k\rangle|^4$ normalised over each row,
a measured product in place of the feed-forward nonlinearity, and a Born readout. Each class probability is then a
ratio of polynomials in the word amplitudes, and its integrated-gradient evidence, with the denominators held, is
computed exactly by a finite quadrature: the **Born evidence per word (BEW)** adds up to the class probability above
its share.

## How it works

Each word $t$ enters as a complex field $a_t\,e_t$ with amplitude $a_t$. A block's attention weighs word $j$ for word
$i$ by

```math
w_{ij} = \frac{|\langle q_i|k_j\rangle|^4}{\sum_{j'} |\langle q_i|k_{j'}\rangle|^4},
```

which needs only overlaps (an interference on light, a Hadamard test on qubits) and a sum. The feed-forward is a
measured product $\mathrm{Re}[(W_1h)\overline{(W_2h)}]$, written back by $W_3$. The readout splits the outcomes into
one group per class $g$ and reads $P_g$; with no information each class gets its share $b_g = 1/3$. With every
denominator held at the input, $P_g - b_g$ is a polynomial $f_g(a)$ of fixed degree, so integrated gradients from
the zero baseline,

```math
e_g(t) = a_t \int_0^1 \frac{\partial f_g}{\partial a_t}(s\,a)\,ds, \qquad \sum_t e_g(t) = P_g - b_g,
```

are computed exactly by Gauss-Legendre quadrature with half the degree in nodes (324 here); the sum holds to about
$10^{-12}$. Red words push towards the answer, blue words against it.

## What the paper finds (e-SNLI, about 98k parameters outside the embedding, 3 seeds)

| | Born Transformer | softmax transformer |
|---|---|---|
| test accuracy | 76.98 +- 0.17 % | 76.62 +- 0.36 % |
| evidence sums to the explained score | to 8e-12 at most | no (integrated gradients, 256 nodes) |
| deleting the word ranked first changes the answer: evidence / attention / random | 45 / 32 / 6 % | 31 / 34 / 5 % |
| top words that annotators highlighted: evidence / attention (chance 0.35) | 0.51 / 0.50 | 0.47 / 0.54 |

The same trained weights compile exactly to passive optics and to qubits. A coherent attention layer with no
measurement before the end is one circuit equal to the model (852 CNOTs on 23 qubits with brickwall maps). On
ibm_kingston, the attention overlaps of one short test pair were read by Hadamard tests and gave the model's answer.

## Contents

| | |
|---|---|
| [`demo/`](demo/) | the two notebooks and their helpers ([`reading_why.py`](demo/reading_why.py), [`ui.py`](demo/ui.py)) |
| [`src/`](src/) | the models, the exact evidence, the trainer, the evaluation (deletion, agreement, attention), the optical and qubit compiles, and the scripts that ran the IBM jobs |
| [`results/`](results/) | the two trained models of the demo (seed 0, 20 epochs) and the vocabulary |
| [`data/`](data/) | where e-SNLI goes (downloaded and checked, not stored); see [`data/README.md`](data/README.md) |
| [`ibm_jobs/`](ibm_jobs/) | the record of every job run on IBM's ibm_kingston: job id, times, calibration, qubits, circuits, raw counts (account details removed) |

Scripts run from the repository root, e.g. `python src/train.py --model Born-p2-ap --blocks 2 --epochs 20 --data
esnli` (after `get_esnli()`), then `python src/faithfulness.py esnli_Born-p2-ap_b2_s0`; they read and write
`results/`.

## Use of AI tools

An AI assistant (Claude, Anthropic) was used throughout this work: to write and run simulation and analysis code, to
prepare and check the hardware jobs, to draw the figures, and to draft and revise the text. All research directions,
decisions and claims are the authors'.

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). e-SNLI: MIT; SNLI sentences and the vocabulary derived from them: CC BY-SA 4.0
(see [`data/README.md`](data/README.md)). Please cite the paper ([`CITATION.cff`](CITATION.cff)); the arXiv entry
follows.
