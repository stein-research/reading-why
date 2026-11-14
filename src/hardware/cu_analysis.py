"""The compute-uncompute runs on ibm_kingston (2026-11-12), from the ledger's counts: each attention weight's
|<k|q>|^2 read as the probability of 0000 after preparing |q> and un-preparing |k> (hw_attention.py --cu).

Per block: the slope and offset fitted to the 32 random calibrations of known |<a|b>|^2, their scatter about the line
against the shot noise at their values, the corrected readings against the ideal |<k|q>|^2 of the same vectors, and
the CZ counts of the circuits as transpiled. Per run: the device's answer and its 95 % shot-noise interval (every
circuit's shots redrawn binomially in P(0000), calibrations refitted; --draws redraws). On both runs also the hybrid
evidence (exact numerators, the measured row sums), its interval (--ev-draws redraws) and the consistency nulls of
device_power.py: unbiased log-normal row errors of the device's own spread.

python cu_analysis.py [--draws 1000] [--ev-draws 500] [--null-draws 100]   -> results/cu_analysis.json   (CPU, minutes)
"""
import json
import math
import pathlib
import re
import sys

import numpy as np
import torch

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
DRAWS, EV_DRAWS, NULL_DRAWS = int(opt("--draws", 1000)), int(opt("--ev-draws", 500)), int(opt("--null-draws", 100))
TAG = "esnli_Born-ap_b2_s0_e20"
RUNS = {"pair 3765": (3765, None, ["5bfe6d8ab641c372", "044c4bbcb3af40f4"]),
        "pair 3763, aliens deleted": (3763, 5, ["6615c7efd68fbb69", "0a2b60fc4c88da98"])}
model, d = load(TAG)
C = d["classes"]
rng = np.random.default_rng(0)


def readings(keys):
    """{block: dict(read={(h, i, j): (P0000, shots)}, cal=[(ideal, P0000, shots)], cz=[...], seconds=..)}."""
    out = {}
    for b, k in enumerate(keys):
        rec = json.load(open(f"ibm_jobs/{k}/record.json"))
        counts = json.load(open(f"ibm_jobs/{k}/counts.json"))
        blk = dict(read={}, cal=[], cz=[], cz_cal=[], job=rec.get("job_id"), key=k,
                   seconds=rec.get("ibm_metrics", {}).get("usage", {}).get("qpu_charge_time_seconds"))
        for c, cnt in zip(rec["circuits"], counts):
            n = sum(cnt.values())
            p = sum(v for kk, v in cnt.items() if set(kk.replace(" ", "")) == {"0"}) / n
            cz = c["ops"].get("cz", 0)
            m = re.match(r"block (\d) head (\d) q(\d+) k(\d+) \|<k\|q>\|\^2 compute-uncompute$", c["note"])
            if m:
                blk["read"][(int(m[2]), int(m[3]), int(m[4]))] = (p, n)
                blk["cz"].append(cz)
                continue
            m = re.match(r"block (\d) random compute-uncompute calibration (\d+) ideal ([.\d]+)$", c["note"])
            assert m, c["note"]
            blk["cal"].append((float(m[3]), p, n))
            blk["cz_cal"].append(cz)
        out[b] = blk
    return out


def fit(cal):
    x, y = np.array([(a, b) for a, b, _ in cal]).T
    slope, off = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    return float(slope), float(off), x, y


def overlap_from(table, draw=None, log=None):
    """The overlap function of compiled_probs: |<k|q>| from the (redrawn) readings, as hw_attention.py --cu does."""
    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        H, L = q.shape[0], q.shape[1]
        t = table[block]
        dr = (lambda p, n: p) if draw is None else (lambda p, n: draw.binomial(n, p) / n)
        slope, off, _, _ = fit([(a, dr(p, n), n) for a, p, n in t["cal"]])
        re_ = torch.zeros(1, H, L, L, dtype=torch.float64)
        for (h, i, j), (p, n) in t["read"].items():
            nq, nk = np.linalg.norm(q[h, i]), np.linalg.norm(k[h, j])
            re_[0, h, i, j] = math.sqrt(max((dr(p, n) - off) / slope, 0.0)) * nq * nk
        if log is not None:
            ov = np.einsum("hie,hje->hij", q.conj(), k)
            ideal, meas = [], []
            for (h, i, j), (p, n) in t["read"].items():
                ideal.append(abs(ov[h, i, j]) ** 2 / (np.linalg.norm(q[h, i]) * np.linalg.norm(k[h, j])) ** 2)
                meas.append(p)
            ideal, meas = np.array(ideal), np.array(meas)
            ex = torch.tensor(np.abs(ov) ** 2)
            hw = re_[0] ** 2
            keep = torch.tensor([(h, i) for h, i, _ in t["read"]])
            rows = torch.ones(H, L, dtype=torch.float64)
            for h, i in set(map(tuple, keep.tolist())):
                rows[h, i] = hw[h, i].sum() / ex[h, i].sum()
            log.append(dict(ideal=ideal, raw=meas, corrected=(meas - off) / slope, rows=rows))
        return re_, torch.zeros_like(re_)
    return overlap


class HwFrozen(Frozen):                       # as hw_attention.HwFrozen: the row sums scaled by measured / exact
    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, dd):
        if not self.replay and dd.dim() == 4:
            dd = dd * self.rows[self.blk][None, :, :, None].to(dd.dtype)
            self.blk += 1
        return super().__call__(dd)


def hw_ev(ids, target, rows):
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    fz = HwFrozen(rows)
    with torch.no_grad():
        f(x0, fz)
    tot = torch.zeros_like(x0)
    for t, w in zip(*nodes01(math.ceil(model.degree() / 2))):
        x = (float(t) * x0).requires_grad_(True)
        (g,) = torch.autograd.grad(f(x, fz.start_replay()).gather(1, target[:, None]).sum(), x)
        tot += float(w) * g
    return (x0 * tot).detach()


out = {}
for name, (idx, dele, keys) in RUNS.items():
    ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
    if dele is not None:
        assert d["words"][int(ids[0, dele])] == "aliens"
        ids[0, dele] = 0
    table = readings(keys)
    log = []
    with torch.no_grad():
        pe = compiled_probs(model, ids)[0][0].numpy()
        pd = compiled_probs(model, ids, overlap=overlap_from(table, log=log))[0][0].numpy()
        ps = np.array([compiled_probs(model, ids, overlap=overlap_from(table, draw=rng))[0][0].numpy()
                       for _ in range(DRAWS)])
    a = int(pe.argmax())
    blocks = []
    for b, t in table.items():
        slope, off, x, y = fit(t["cal"])
        res = y - slope * x - off
        shot = float(np.sqrt(np.mean(y.clip(1e-9, 1) * (1 - y.clip(0, 1)) / t["cal"][0][2])))
        lg = log[b]
        blocks.append(dict(job=t["job"], ledger=t["key"], seconds=t["seconds"], readings=len(t["read"]),
                           calibrations=len(t["cal"]), slope=slope, offset=off,
                           scatter=float(np.sqrt((res ** 2).sum() / (len(res) - 2))), shot=shot,
                           corr_raw_ideal=float(np.corrcoef(lg["raw"], lg["ideal"])[0, 1]),
                           mae_corrected=float(np.abs(lg["corrected"] - lg["ideal"]).mean()),
                           cz_median=float(np.median(t["cz"])), cz_max=int(max(t["cz"])),
                           cz_cal_median=float(np.median(t["cz_cal"])),
                           row_log_sd=float(np.log(lg["rows"][lg["rows"] != 1.0].numpy()).std())))
    lo, hi = np.percentile(ps[:, a], [2.5, 97.5])
    o = dict(index=idx, deleted=dele, exact=C[a], exact_p=pe.tolist(), device=C[int(pd.argmax())], device_p=pd.tolist(),
             ci95=[float(lo), float(hi)], same_answer_share=float((ps.argmax(1) == a).mean()), draws=DRAWS,
             blocks=blocks)
    if True:                                           # the evidence with the measured row sums (both runs)
        tok = np.where(ids[0].numpy() != 0)[0]
        with torch.no_grad():
            target = model(ids).argmax(1)
        ex = evidence(model, ids, target)[0][0, tok].detach().numpy()
        corr = lambda e: float(np.corrcoef(ex, e)[0, 1])
        rows = [lg["rows"] for lg in log]
        ev = hw_ev(ids, target, rows)[0, tok].numpy()
        evs = []
        for _ in range(EV_DRAWS):
            lg2 = []
            with torch.no_grad():
                compiled_probs(model, ids, overlap=overlap_from(table, draw=rng, log=lg2))
            evs.append(hw_ev(ids, target, [l["rows"] for l in lg2])[0, tok].numpy())
        evs = np.array(evs)
        cs = np.array([corr(e) for e in evs])
        sd = float(np.mean([np.log(r[r != 1.0].numpy()).std() for r in rows]))
        nulls = np.array([corr(hw_ev(ids, target, [torch.where(r != 1.0, torch.tensor(np.exp(rng.normal(0, sd, r.shape))), r)
                                                   for r in rows])[0, tok].numpy()) for _ in range(NULL_DRAWS)])
        words = [d["words"][int(w)] for w in ids[0, tok]]
        o["evidence"] = dict(words=words, exact=ex.tolist(), device=ev.tolist(), corr=corr(ev),
                             corr_ci95=np.percentile(cs, [2.5, 97.5]).tolist(),
                             ev_ci95_low=np.percentile(evs, 2.5, 0).tolist(), ev_ci95_high=np.percentile(evs, 97.5, 0).tolist(),
                             signs_right=bool((np.sign(ev) == np.sign(ex)).all()),
                             top_word_same_share=float((evs.argmax(1) == ex.argmax()).mean()),
                             sum_device=float(ev.sum()), row_log_sd=sd,
                             random_rows_median=float(np.median(nulls)),
                             random_rows_share_below_device=float((nulls < corr(ev)).mean()))
    out[name] = o
    print(name, json.dumps({k: v for k, v in o.items() if k != "evidence"}, default=float), flush=True)
    if "evidence" in o:
        print("  evidence", json.dumps(o["evidence"]), flush=True)
json.dump(out, open("results/cu_analysis.json", "w"), indent=1)
