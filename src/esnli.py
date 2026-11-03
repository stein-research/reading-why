"""e-SNLI as padded token ids: premise <sep> hypothesis -> entailment / neutral / contradiction, with the words the
annotators highlighted as ground truth for the evidence.

Data: github.com/OanaMariaCamburu/e-SNLI at commit 7b585a3f (dataset/*.csv; MIT licence for the repository, the SNLI
sentences under CC BY-SA 4.0, Bowman et al. 2015; Camburu et al. 2018); checksums in data/esnli/SHA256SUMS.
Tokens as data.py (lower-cased words, any other character its own token). Ground truth: a token is highlighted if
it comes from a word one annotator starred in Sentence1_marked_1 / Sentence2_marked_1 ("*w*", punctuation attached
to a word goes with it); dev and test have three annotators and take the union of them. The vocabulary is built from
the training split: id 0 padding, 1 <unk>, 2 <sep>, then words seen at least `min_count` times. Pairs longer than
`max_len` are cut at the end (the hypothesis loses its tail).
"""
import collections
import csv
import hashlib
import os
import urllib.request

import numpy as np

import data as banking

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "esnli")
LABELS = ["entailment", "neutral", "contradiction"]
PAD, UNK, SEP = 0, 1, 2


def marked(text):
    """Tokens of a starred sentence and, per token, whether its word was starred."""
    toks, flags = [], []
    for chunk in text.split():
        star = chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 1
        t = banking.tokenize(chunk.strip("*") if star else chunk)
        toks += t
        flags += [star] * len(t)
    return toks, flags


URL = "https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/7b585a3f/dataset/"   # the paper's commit
SHA256 = {                                                     # data/SHA256SUMS
    "esnli_train_1.csv": "7311c7bc16ad9f6a9adcd116a62ef991e1803dc0d71e253ce66b975c2aba8ee5",
    "esnli_train_2.csv": "c8827fa2ba1ef5891f077d6fafbfc2fdb6cab1271968b74257e08d3b2b3cbacc",
    "esnli_dev.csv": "d7a656d196eea18fd827d6f5486b00f5d4cf469dbf13a252cc211495a186bcb4",
    "esnli_test.csv": "9269aa8075dbfed0fa8a5012eb2ac9c18a92be139671f05df897f69092592927",
}


def fetch(files=tuple(SHA256)):
    """Download the e-SNLI files that are not yet in data/esnli (about 200 MB in all), each checked by its SHA-256."""
    os.makedirs(DATA, exist_ok=True)
    for name in files:
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            continue
        print(f"downloading {name} from the e-SNLI repository ...", flush=True)
        blob = urllib.request.urlopen(URL + name, timeout=300).read()
        if hashlib.sha256(blob).hexdigest() != SHA256[name]:
            raise RuntimeError(f"{name}: unexpected SHA-256, not used")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"{name}: {len(blob) / 1e6:.0f} MB", flush=True)


def read(split):
    files = ["esnli_train_1.csv", "esnli_train_2.csv"] if split == "train" else [f"esnli_{split}.csv"]
    fetch(files)                                               # downloads what is missing, once
    out = []
    for fn in files:
        with open(os.path.join(DATA, fn), newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["gold_label"] not in LABELS:
                    continue
                p, gp = marked(r["Sentence1_marked_1"])
                h, gh = marked(r["Sentence2_marked_1"])
                for a in (2, 3):
                    if r.get(f"Sentence1_marked_{a}"):
                        p2, g2 = marked(r[f"Sentence1_marked_{a}"])
                        h2, h2g = marked(r[f"Sentence2_marked_{a}"])
                        if p2 == p and h2 == h:
                            gp = [x or y for x, y in zip(gp, g2)]
                            gh = [x or y for x, y in zip(gh, h2g)]
                out.append((p, h, gp, gh, LABELS.index(r["gold_label"])))
    return out


def load(max_len=64, min_count=3, n_train=None):
    tr, te = read("train"), read("test")
    if n_train:
        tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:n_train]]
    counts = collections.Counter(w for p, h, *_ in tr for w in p + h)
    words = ["<pad>", "<unk>", "<sep>"] + sorted((w for w, c in counts.items() if c >= min_count),
                                                key=lambda w: (-counts[w], w))
    wid = {w: i for i, w in enumerate(words)}

    def encode(rows):
        x = np.full((len(rows), max_len), PAD, dtype=np.int64)
        gt = np.zeros((len(rows), max_len), dtype=bool)
        y = np.array([r[4] for r in rows], dtype=np.int64)
        for i, (p, h, gp, gh, _) in enumerate(rows):
            ids = ([wid.get(w, UNK) for w in p] + [SEP] + [wid.get(w, UNK) for w in h])[:max_len]
            g = (gp + [False] + gh)[:max_len]
            x[i, : len(ids)], gt[i, : len(g)] = ids, g
        return x, y, gt

    x_tr, y_tr, g_tr = encode(tr)
    x_te, y_te, g_te = encode(te)
    x_dv, y_dv, g_dv = encode(read("dev"))      # for choosing a learning rate or length; test only reports
    return dict(x_train=x_tr, y_train=y_tr, gt_train=g_tr, x_test=x_te, y_test=y_te, gt_test=g_te,
                x_dev=x_dv, y_dev=y_dv, gt_dev=g_dv,
                words=words, classes=LABELS, n_distinct=len(counts))


if __name__ == "__main__":
    d = load()
    for s in ("train", "dev", "test"):
        x, g = d[f"x_{s}"], d[f"gt_{s}"]
        n = (x != PAD).sum(1)
        print(f"{s}: {len(x)} pairs, tokens median {int(np.median(n))} max {n.max()}, cut at 64: {(n == 64).mean():.4f}, "
              f"<unk> {(x == UNK).sum() / n.sum():.4f}, highlighted per pair {g.sum(1).mean():.2f} "
              f"(none: {(g.sum(1) == 0).mean():.3f}), labels {np.bincount(d[f'y_{s}'])}")
    print(f"distinct training words {d['n_distinct']}, vocabulary {len(d['words'])}")
    inv = d["words"]
    for i in range(3):
        print(LABELS[d["y_test"][i]], "|", " ".join(inv[t].upper() if gg else inv[t]
                                                     for t, gg in zip(d["x_test"][i], d["gt_test"][i]) if t))
