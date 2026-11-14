"""ORDER: a generated task in which word order decides the class, with the deciding words known.

Each query asks to move money between two of eight places, e.g. "can you please move money now from savings later
to wallet". The class is the ordered pair (source, destination): 8 x 7 = 56 classes. The two phrases "from X" and
"to Y" come in either order, so position alone does not give the direction; half the queries add a decoy phrase
"about Z" with a third place, so the words present do not tell which two places count. Fillers (0-2 words, never a
place or a preposition) go between the parts. A bag of words therefore cannot know the direction, and with a decoy
not even the pair; `bow_baseline.py --data order` measures how far it gets.

Ground truth: the deciding tokens are X, Y and their prepositions "from" and "to" (four per query); the decoy and
its "about", the opening and the fillers decide nothing. Generated with a fixed seed: nothing downloaded, no licence.
"""
import numpy as np

PLACES = ["card", "account", "savings", "wallet", "loan", "deposit", "pocket", "vault"]
OPENINGS = [["move", "money"], ["can", "you", "move", "money"], ["please", "transfer", "cash"], ["send", "funds"],
            ["i", "want", "a", "transfer"], ["shift", "my", "balance"], ["transfer", "everything"]]
FILLERS = ["please", "now", "today", "quickly", "later", "again", "my", "the", "some", "all", "money", "funds", "asap",
           "thanks", "kindly", "right", "away", "soon", "then", "also"]
PAIRS = [(a, b) for a in range(8) for b in range(8) if a != b]
PAD, UNK = 0, 1


def _query(rng):
    pair = PAIRS[rng.integers(len(PAIRS))]
    src, dst = PLACES[pair[0]], PLACES[pair[1]]
    parts = [[("from", 1), (src, 1)], [("to", 1), (dst, 1)]]
    if rng.random() < 0.5:
        parts = parts[::-1]
    if rng.random() < 0.5:
        z = PLACES[rng.choice([i for i in range(8) if i not in pair])]
        parts.insert(rng.integers(len(parts) + 1), [("about", 0), (z, 0)])
    out = [(w, 0) for w in OPENINGS[rng.integers(len(OPENINGS))]]
    for p in parts:
        out += [(FILLERS[rng.integers(len(FILLERS))], 0) for _ in range(rng.integers(3))]
        out += p
    out += [(FILLERS[rng.integers(len(FILLERS))], 0) for _ in range(rng.integers(3))]
    return out, PAIRS.index(pair)


def load(n_train=10000, n_test=3080, max_len=32, seed=0):
    words = ["<pad>", "<unk>"] + sorted(set(PLACES + FILLERS + ["from", "to", "about"] + sum(OPENINGS, [])))
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.default_rng(seed)

    def make(n):
        x = np.full((n, max_len), PAD, dtype=np.int64)
        gt = np.zeros((n, max_len), dtype=bool)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            q, y[i] = _query(rng)
            assert len(q) <= max_len
            x[i, : len(q)] = [wid[w] for w, _ in q]
            gt[i, : len(q)] = [g for _, g in q]
        return x, y, gt

    x_tr, y_tr, g_tr = make(n_train)
    x_te, y_te, g_te = make(n_test)
    classes = [f"{PLACES[a]}->{PLACES[b]}" for a, b in PAIRS]
    return dict(x_train=x_tr, y_train=y_tr, gt_train=g_tr, x_test=x_te, y_test=y_te, gt_test=g_te,
                words=words, classes=classes)


if __name__ == "__main__":
    d = load()
    inv = d["words"]
    for i in range(4):
        q = [inv[t] for t in d["x_train"][i] if t]
        print(d["classes"][d["y_train"][i]], "|", " ".join(w.upper() if g else w for w, g in zip(q, d["gt_train"][i])))
    n = (d["x_train"] != 0).sum(1)
    tr = {tuple(r) for r in d["x_train"]}
    print(f"tokens median {int(np.median(n))} max {n.max()}; test queries also in train "
          f"{np.mean([tuple(r) in tr for r in d['x_test']]):.4f}; vocabulary {len(inv)}")


# ------------------------------------------------------------------------------------------------ associative recall
RECALL_KEYS, RECALL_VALUES = 32, 16


def load_recall(n_pairs=8, n_train=100000, n_test=3080, max_len=32, seed=0):
    """RECALL: "k3 v7 k1 v2 k8 v5 ... ? k8" -> v5. The context holds `n_pairs` key-value pairs with distinct keys
    (32 keys, 16 values, values may repeat); after "?" one key of the context is asked; the class is its value
    (16 classes). Only content-based lookup solves it: find the context key equal to the asked one, read the token
    after it. A bag of words sees which keys and values occur, and the asked key twice, but not which value goes
    with which key.

    Ground truth: the asked key, the matching context key, its value and "?" (four per query)."""
    assert 2 * n_pairs + 2 <= max_len
    words = ["<pad>", "<unk>", "?"] + [f"k{i}" for i in range(RECALL_KEYS)] + [f"v{i}" for i in range(RECALL_VALUES)]
    wid = {w: i for i, w in enumerate(words)}
    rng = np.random.default_rng(seed + 1000 * n_pairs)

    def make(n):
        x = np.full((n, max_len), PAD, dtype=np.int64)
        gt = np.zeros((n, max_len), dtype=bool)
        y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            keys = rng.choice(RECALL_KEYS, n_pairs, replace=False)
            vals = rng.integers(RECALL_VALUES, size=n_pairs)
            j = rng.integers(n_pairs)
            toks = sum(([f"k{k}", f"v{v}"] for k, v in zip(keys, vals)), []) + ["?", f"k{keys[j]}"]
            x[i, : len(toks)] = [wid[t] for t in toks]
            gt[i, [2 * j, 2 * j + 1, len(toks) - 2, len(toks) - 1]] = True
            y[i] = vals[j]
        return x, y, gt

    x_tr, y_tr, g_tr = make(n_train)
    x_te, y_te, g_te = make(n_test)
    return dict(x_train=x_tr, y_train=y_tr, gt_train=g_tr, x_test=x_te, y_test=y_te, gt_test=g_te,
                words=words, classes=[f"v{i}" for i in range(RECALL_VALUES)])


def get(name):
    """Every task by name: banking77, order, recall<n> (n key-value pairs), esnli."""
    if name == "banking77":
        import data
        return data.load()
    if name == "order":
        return load()
    if name == "esnli":
        import esnli
        return esnli.load()
    if name.startswith("recall"):
        return load_recall(int(name[6:]))
    raise ValueError(name)
