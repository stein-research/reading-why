"""Hadamard tests against compute-uncompute on the same quantity (2026-11-12): the 128 Born weights of pair 3765's first
block, whose vectors are the same in both runs (ledger a1d6be83e81a3267 and 5bfe6d8ab641c372): each method's
calibrated |<k|q>|^2 of the normalised states against the ideal; mean absolute error, RMSE, bias, correlation.
python hw_cu_compare.py   (CPU, a minute) -> results/hw_cu_compare.json (printed lines)
"""
import builtins as _b, json as _j
_LINES = []
_print = _b.print
def print(*a, **k):
    _LINES.append(" ".join(str(x) for x in a)); _print(*a, **k)
import json, re, sys, numpy as np, torch
from compile_hw import compiled_probs, load
model, d = load("esnli_Born-ap_b2_s0_e20")
ids = torch.tensor(d["x_test"][3765:3766]).clone()
cap = {}
def ov(qi, kj, block):
    q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
    o = np.einsum("hie,hje->hij", q.conj(), k)
    cap[block] = (q, k)
    t = torch.tensor(o)[None]
    return t.real.clone(), t.imag.clone()
with torch.no_grad():
    p = compiled_probs(model, ids, overlap=ov)[0][0].numpy()
print("p exact via capture", p)
q, k = cap[0]
nq = np.linalg.norm(q, axis=-1); nk = np.linalg.norm(k, axis=-1)
ideal = np.abs(np.einsum("hie,hje->hij", q.conj(), k))**2 / (nq[:, :, None]*nk[:, None, :])**2
def load_job(key):
    r = json.load(open(f"ibm_jobs/{key}/record.json")); c = json.load(open(f"ibm_jobs/{key}/counts.json"))
    return r["circuits"], c
def fit(x, y):
    return np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
# HT block 0 of 3765
circ, cnts = load_job("a1d6be83e81a3267")
reim = {}; cal = []
for ci, cn in zip(circ, cnts):
    n = sum(cn.values()); z = sum(v for kk, v in cn.items() if kk.replace(' ', '')[-1] == '0')  # ancilla bit
    keys = list(cn.keys())
    p0 = z / n
    m = re.match(r"block 0 head (\d) q(\d+) k(\d+) (Re|Im)$", ci["note"])
    if m: reim[(int(m[1]), int(m[2]), int(m[3]), m[4])] = 2*p0-1; continue
    m = re.match(r"block 0 random calibration \d+ ideal ([-+.\d]+)$", ci["note"])
    if m: cal.append((float(m[1]), 2*p0-1))
print("HT bitstring example", keys[:3])
x, y = np.array(cal).T; s, o = fit(x, y); print("HT slope off", s, o, "resid", np.std(y - s*x - o))
ht, idl = [], []
for (h, i, j, part), v in reim.items():
    if part != 'Re': continue
    re_ = (v - o)/s; im_ = (reim[(h, i, j, 'Im')] - o)/s
    ht.append(re_**2 + im_**2); idl.append(ideal[h, i, j])
ht, idl = np.array(ht), np.array(idl)
# CU block 0
circ, cnts = load_job("5bfe6d8ab641c372")
cu, cal = {}, []
for ci, cn in zip(circ, cnts):
    n = sum(cn.values()); p0 = sum(v for kk, v in cn.items() if set(kk.replace(' ', '')) == {'0'})/n
    m = re.match(r"block 0 head (\d) q(\d+) k(\d+) ", ci["note"])
    if m: cu[(int(m[1]), int(m[2]), int(m[3]))] = p0; continue
    m = re.match(r".*calibration \d+ ideal ([.\d]+)$", ci["note"]); cal.append((float(m[1]), p0))
x, y = np.array(cal).T; s2, o2 = fit(x, y)
print("CU cal ideal quantiles", np.percentile(x, [0, 25, 50, 75, 100]).round(3))
cuv = np.array([(cu[kk]-o2)/s2 for kk in sorted(cu)]); cid = np.array([ideal[kk] for kk in sorted(cu)])
print("reading ideal quantiles", np.percentile(cid, [0, 25, 50, 75, 100]).round(3))
for name, a, b in [("HT", ht, idl), ("CU", cuv, cid)]:
    e = a - b
    print(name, "n", len(a), "MAE", abs(e).mean().round(4), "RMSE", np.sqrt((e**2).mean()).round(4), "bias", e.mean().round(4), "corr", np.corrcoef(a, b)[0, 1].round(4))
kk_ht=[kk[:3] for kk in reim if kk[3]=="Re"]; eh=dict(zip(kk_ht, ht-idl)); ec=dict(zip(sorted(cu), cuv-cid)); ks=sorted(ec); print("err corr HT vs CU same pairs", np.corrcoef([eh[k] for k in ks],[ec[k] for k in ks])[0,1].round(3)); rel=np.array([ec[k] for k in ks]); print("CU err vs ideal corr", np.corrcoef(rel, cid)[0,1].round(3))
_j.dump(dict(script="hw_cu_compare.py", lines=_LINES), open("results/hw_cu_compare.json", "w"), indent=1)
