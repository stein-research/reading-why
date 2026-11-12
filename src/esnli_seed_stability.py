"""How stable BEW is across training seeds (referee round 7, 2026-11-12): on faithfulness.py's 1,000 e-SNLI test
pairs, where the three seeds agree on the answer, the seed-to-seed Pearson and Spearman correlation of the per-word
evidence and the share with the same top word, for BEW and for the softmax model's integrated gradients; and BEW
against its own leave-one-out.
python esnli_seed_stability.py   (GPU, ~10 min) -> results/esnli_seed_stability.json (printed lines)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)
import json, numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
import synth
from evidence import evidence
from models import build
dev = "cuda"
d = synth.get("esnli"); V, C = len(d["words"]), len(d["classes"])
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
ids = torch.tensor(d["x_test"][pick], device=dev); present = ids != 0; L = ids.shape[1]
def load(tag):
    cfg = json.load(open(f"results/{tag}.json")); m = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=L).to(dev)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev)); return m.double().eval()
res = {}
for fam in ("Born-ap", "Softmax", "Poly-sim-born"):
    ms = [load(f"esnli_{fam}_b2_s{s}_e20") for s in range(3)]
    prob = lambda m, i: m(i) if m.family == "B" else F.softmax(m(i), -1)
    with torch.no_grad(): P = [prob(m, ids) for m in ms]
    pred = torch.stack([p.argmax(1) for p in P]); agree = (pred[0] == pred[1]) & (pred[0] == pred[2]); target = pred[0]
    print(f"{fam}: all three seeds give the same answer on {agree.double().mean().item():.3f} of pairs")
    EV, LOO = [], []
    for m, p0 in zip(ms, P):
        ev = torch.cat([evidence(m, ids[i:i+100], target[i:i+100])[0] for i in range(0, 1000, 100)]); EV.append(ev.detach().cpu().numpy())
        loo = torch.zeros(ids.shape, dtype=torch.double, device=dev)
        with torch.no_grad():
            for t in range(L):
                x = ids.clone(); x[:, t] = 0; loo[:, t] = (p0 - prob(m, x)).gather(1, target[:, None])[:, 0]
        LOO.append(loo.cpu().numpy())
    pr = present.cpu().numpy(); ag = agree.cpu().numpy()
    for name, X in (("evidence", EV), ("leave-one-out", LOO)):
        for (a, b) in ((0, 1), (0, 2), (1, 2)):
            rs, pe, top = [], [], []
            for i in np.where(ag)[0]:
                k = pr[i]
                if k.sum() < 4: continue
                rs.append(spearmanr(X[a][i, k], X[b][i, k])[0]); pe.append(np.corrcoef(X[a][i, k], X[b][i, k])[0, 1]); top.append(X[a][i, k].argmax() == X[b][i, k].argmax())
            print(f"   {name:14s} seeds {a},{b}: median Spearman {np.nanmedian(rs):.3f}  median Pearson {np.nanmedian(pe):.3f}  same top word {np.mean(top):.3f}  (n={len(rs)})", flush=True)
    # within-model: evidence vs LOO
    for s in range(3):
        rs = [spearmanr(EV[s][i, pr[i]], LOO[s][i, pr[i]])[0] for i in np.where(ag)[0] if pr[i].sum() >= 4]
        print(f"   seed {s}: evidence vs its own leave-one-out, median Spearman {np.nanmedian(rs):.3f}")

_j.dump(dict(script="esnli_seed_stability.py", lines=_LINES), open("results/esnli_seed_stability.json", "w"), indent=1)
