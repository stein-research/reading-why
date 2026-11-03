# The notebooks on your own computer

The two notebooks run in Colab with no install (links in the [main README](../README.md)). To run them locally:

```bash
git clone https://github.com/stein-research/reading-why
cd reading-why
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy torch matplotlib ipywidgets jupyterlab
jupyter lab demo/demo.ipynb
```

- **`demo.ipynb`** runs on a CPU (a few seconds a pair). The first run downloads the e-SNLI test split (7 MB) into
  `data/esnli/` and checks its SHA-256.
- **`train.ipynb`** wants a GPU (an NVIDIA card with CUDA-enabled PyTorch). It downloads all of e-SNLI (about 200 MB,
  each file checked). The default is 5 epochs, about 15 min for the Born Transformer on a laptop GPU; the paper
  trains 20.

## Without notebooks

`reading_why.py` is the demo's engine and works from plain Python (run from `demo/`, or put `demo/` on the path):

```python
import reading_why as rw
m = rw.load("born")                                   # or "softmax", or the tag of a run in results/
r = rw.explain(m, "A woman working long hours.", "A woman is sleeping.")
print(r["answer"], r["probs"])                        # contradiction, P_g of the three classes
for w, e, a in zip(r["words"], r["evidence"], r["attention"]):
    print(f"{w:>10s}  evidence {e:+.4f}  attention {a:.3f}")
print(r["evidence"].sum(), r["explained"])            # equal: the evidence sums to P_g - b_g

ids, words = rw.without(r, [2, 4])                    # delete words 2 and 4 (in place, as in the paper)
print(rw.explain(m, ids=ids, toks=words)["probs"])
```

| file | what |
|---|---|
| `reading_why.py` | load a model, encode a pair, the answer, the Born evidence per word (exact quadrature, nodes run as one batch), the attention each word receives, deletion, the e-SNLI download |
| `ui.py` | the widgets and the HTML card (`ui.demo()`; `ui.show(model, premise, hypothesis)` without widgets) |
| `demo.ipynb`, `train.ipynb` | the two notebooks |
