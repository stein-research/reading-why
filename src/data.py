"""BANKING77 as padded token ids.

Words are lower-cased runs of letters, digits and apostrophes; any other non-space character is a token of its own.
The vocabulary is built from the training split only: id 0 is padding (an absent position, amplitude zero in the
Born models), id 1 is <unk>, then the most frequent training words up to `vocab` ids in all (ties broken
alphabetically). Queries longer than `max_len` tokens are cut at the end.
"""
import collections
import csv
import json
import os
import re

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "banking77")
TOKEN = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")
PAD, UNK = 0, 1


def tokenize(text):
    return TOKEN.findall(text.lower())


def read_split(name):
    with open(os.path.join(DATA, f"{name}.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["category"] for r in rows]


def load(vocab=4096, max_len=32):
    """Returns dict with x_train, y_train, x_test, y_test (int64 arrays, x padded with PAD), words, classes."""
    with open(os.path.join(DATA, "categories.json")) as f:
        classes = json.load(f)
    cid = {c: i for i, c in enumerate(classes)}
    tr_text, tr_cat = read_split("train")
    te_text, te_cat = read_split("test")
    counts = collections.Counter(w for t in tr_text for w in tokenize(t))
    words = ["<pad>", "<unk>"] + [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))][: vocab - 2]
    wid = {w: i for i, w in enumerate(words)}

    def encode(texts):
        x = np.full((len(texts), max_len), PAD, dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [wid.get(w, UNK) for w in tokenize(t)][:max_len]
            x[i, : len(ids)] = ids
        return x

    return dict(x_train=encode(tr_text), y_train=np.array([cid[c] for c in tr_cat]),
                x_test=encode(te_text), y_test=np.array([cid[c] for c in te_cat]),
                words=words, classes=classes, n_distinct=len(counts))


if __name__ == "__main__":
    d = load()
    for s in ("train", "test"):
        x = d[f"x_{s}"]
        n = (x != PAD).sum(1)
        print(f"{s}: {len(x)} queries, tokens median {int(np.median(n))}, cut at 32: {(n == 32).mean():.3f}, "
              f"<unk> share {(x == UNK).sum() / n.sum():.4f}")
    print(f"distinct training words {d['n_distinct']}, vocabulary {len(d['words'])}")
