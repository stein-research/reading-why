"""Reading Why, the demo's engine: the trained models, a sentence pair in, the answer, the Born evidence per word (BEW)
and the attention each word receives out.

    import reading_why as rw
    m = rw.load("born")                          # or "softmax"
    r = rw.explain(m, "A woman working long hours.", "A woman is sleeping.")
    r["answer"], r["words"], r["evidence"], r["attention"]

The Born Transformer's evidence is exact: integrated gradients of its class probability with the denominators held,
by a Gauss-Legendre rule that is exact for that polynomial; the words' evidence sums to P_g - b_g (b_g = |S_g| / 256, about 1/3, the
share of a class when nothing is known). The softmax model's is ordinary integrated gradients (256 nodes) of its
logit, which does not sum exactly. Attention: the weight each word receives, averaged over the two blocks, the heads
and the words that look at it.
"""
import csv
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data as _tok                     # noqa: E402  (the tokenizer)
import esnli                            # noqa: E402
import math                             # noqa: E402
from evidence import _score_fn, nodes01  # noqa: E402
from models import Frozen, build        # noqa: E402

LABELS = esnli.LABELS
MAX_LEN = 64
RUNS = {"born": "esnli_Born-p2-ap_b2_s0_e20", "softmax": "esnli_Softmax_b2_s0_e20"}
# e-SNLI test split, at the commit the paper used (MIT; the SNLI sentences CC BY-SA 4.0)
TEST_URL = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/7b585a3f/dataset/esnli_test.csv"
TEST_SHA256 = "9269aa8075dbfed0fa8a5012eb2ac9c18a92be139671f05df897f69092592927"

WORDS = json.load(open(os.path.join(ROOT, "results", "esnli_vocab.json")))
WID = {w: i for i, w in enumerate(WORDS)}
torch.set_grad_enabled(True)


def load(name="born"):
    """A trained model: "born" or "softmax" (the paper's seed-0 runs), or the tag of a run in results/."""
    tag = RUNS.get(name, name)
    cfg = json.load(open(os.path.join(ROOT, "results", f"{tag}.json")))
    m = build(cfg["model"], len(WORDS), len(LABELS), blocks=cfg["blocks"], max_len=MAX_LEN)
    m.load_state_dict(torch.load(os.path.join(ROOT, "results", f"{tag}.pt"), map_location="cpu"))
    m.double().eval()
    m.name, m.test_acc = {"B": "born", "S": "softmax", "P1": "poly"}[m.family], cfg["test_acc"]
    return m


def encode(premise, hypothesis):
    """(ids [1, n], tokens, known): premise <sep> hypothesis, lower-cased, at most 64 tokens (no padding: the models
    give the same answer padded or not, and are faster without); a word outside the vocabulary is <unk>."""
    toks = _tok.tokenize(premise) + ["<sep>"] + _tok.tokenize(hypothesis)
    toks = toks[:MAX_LEN]
    ids = torch.tensor([[WID.get(t, esnli.UNK) for t in toks]])
    return ids, toks, [t in WID for t in toks]


def probs(model, ids):
    with torch.no_grad():
        out = model(ids)
    return out if model.family == "B" else torch.softmax(out, -1)


def attention(model, ids):
    """Weight each word receives, [n]: mean over blocks, heads and the present words that look at it."""
    rec = []
    if model.family == "S":
        orig = torch.nn.MultiheadAttention.forward

        def fwd(self, *a, **kw):
            kw.update(need_weights=True, average_attn_weights=False)
            out = orig(self, *a, **kw)
            rec.append(out[1])
            return out
        torch.nn.MultiheadAttention.forward = fwd
        try:
            probs(model, ids)
        finally:
            torch.nn.MultiheadAttention.forward = orig
    else:
        orig = torch.einsum

        def ein(eq, *ops):
            if eq == "bhij,bjhe->bihe":
                rec.append(ops[0].real if ops[0].is_complex() else ops[0])
            return orig(eq, *ops)
        torch.einsum = ein
        try:
            probs(model, ids)
        finally:
            torch.einsum = orig
    L = ids.shape[1]
    q = (ids != 0)[:, None, :, None].double()
    recv = [((w[:, :, :L, :L].double() * q).sum(2) / q.sum(2).clamp(min=1)).mean(1) for w in rec]
    return (sum(recv) / len(recv))[0].detach().numpy()


def evidence(model, ids, target, chunk=64):
    """evidence.evidence (src/evidence.py) for one pair, with the quadrature nodes run as one batch instead of one
    by one: the same numbers (the denominators held at the explained input, now broadcast over the nodes), many
    times faster on a CPU. Returns (evidence [n], f(x), f(0))."""
    x0, _ = _score_fn(model, ids)
    x0 = x0.detach()
    deg = model.degree() if hasattr(model, "degree") else None
    n_nodes = math.ceil(deg / 2) if deg else 256
    fz = Frozen()
    _, f1 = _score_fn(model, ids)
    with torch.no_grad():
        full = f1(x0, fz)[0, target].item()
        zero = f1(torch.zeros_like(x0), fz.start_replay())[0, target].item()
    t, w = nodes01(n_nodes)
    total = torch.zeros_like(x0[0])
    for i in range(0, n_nodes, chunk):
        tt = torch.tensor(t[i:i + chunk], dtype=x0.real.dtype)
        b = len(tt)
        _, fb = _score_fn(model, ids.expand(b, -1))
        x = (tt.view(-1, *[1] * (x0.dim() - 1)) * x0.expand(b, *x0.shape[1:])).requires_grad_(True)
        out = fb(x, fz.start_replay())[:, target].sum()
        (g,) = torch.autograd.grad(out, x)
        total += (torch.tensor(w[i:i + chunk], dtype=g.real.dtype).view(-1, *[1] * (g.dim() - 1)) * g).sum(0)
    ig = x0[0] * total
    return (ig if ig.dim() == 1 else ig.sum(-1)), full, zero


def explain(model, premise=None, hypothesis=None, ids=None, toks=None):
    """The answer, the class probabilities, each word's evidence for the answer and the attention it receives."""
    if ids is None:
        ids, toks, _ = encode(premise, hypothesis)
    n = ids.shape[1]
    p = probs(model, ids)[0]
    t = int(p.argmax())
    ev, full, zero = evidence(model, ids, t)
    e = ev[:n].detach().numpy()
    explained = full - zero
    return dict(model=model.name, words=list(toks[:n]), probs=p.numpy(), answer=LABELS[t], target=t,
                evidence=e, explained=explained, gap=abs(e.sum() - explained),
                attention=attention(model, ids), ids=ids)


def without(r, pos):
    """The ids of an explained pair with the word at `pos` (an index, or a list of them) removed (amplitude 0 /
    token absent), and its words."""
    ids = r["ids"].clone()
    ids[0, list(pos) if isinstance(pos, (list, tuple, set)) else pos] = 0   # the slots stay, empty (as in the paper)
    return ids, r["words"]


ESNLI_FILES = {   # the e-SNLI files at the paper's commit, with their SHA-256 (data/SHA256SUMS)
    "esnli_train_1.csv": "7311c7bc16ad9f6a9adcd116a62ef991e1803dc0d71e253ce66b975c2aba8ee5",
    "esnli_train_2.csv": "c8827fa2ba1ef5891f077d6fafbfc2fdb6cab1271968b74257e08d3b2b3cbacc",
    "esnli_dev.csv": "d7a656d196eea18fd827d6f5486b00f5d4cf469dbf13a252cc211495a186bcb4",
    "esnli_test.csv": TEST_SHA256,
}


def get_esnli(files=ESNLI_FILES):
    """Download the e-SNLI files that are not yet in data/esnli (about 200 MB in all), each checked by its SHA-256."""
    os.makedirs(esnli.DATA, exist_ok=True)
    for name, sha in files.items():
        path = os.path.join(esnli.DATA, name)
        if os.path.exists(path):
            continue
        blob = urllib.request.urlopen(TEST_URL.rsplit("/", 1)[0] + "/" + name, timeout=300).read()
        if hashlib.sha256(blob).hexdigest() != sha:
            raise RuntimeError(f"{name}: unexpected SHA-256, not used")
        open(path, "wb").write(blob)
        print(f"{name}: {len(blob) / 1e6:.0f} MB")


def test_pairs():
    """The e-SNLI test split: [(premise, hypothesis, label, premise words marked, hypothesis words marked)]. Reads
    data/esnli/esnli_test.csv, downloading it first (7 MB, checked by its SHA-256) when it is not there."""
    get_esnli({"esnli_test.csv": TEST_SHA256})
    path = os.path.join(esnli.DATA, "esnli_test.csv")
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["gold_label"] not in LABELS:
                continue
            _, gp = esnli.marked(r["Sentence1_marked_1"])
            _, gh = esnli.marked(r["Sentence2_marked_1"])
            for a in (2, 3):
                if r.get(f"Sentence1_marked_{a}"):
                    p2, g2 = esnli.marked(r[f"Sentence1_marked_{a}"])
                    h2, h2g = esnli.marked(r[f"Sentence2_marked_{a}"])
                    if len(g2) == len(gp) and len(h2g) == len(gh):
                        gp = [x or y for x, y in zip(gp, g2)]
                        gh = [x or y for x, y in zip(gh, h2g)]
            out.append((r["Sentence1"].strip(), r["Sentence2"].strip(), r["gold_label"], gp, gh))
    return out


def marked_mask(pair):
    """Per token of premise <sep> hypothesis, whether an annotator highlighted it."""
    return (list(pair[3]) + [False] + list(pair[4]))[:MAX_LEN]
