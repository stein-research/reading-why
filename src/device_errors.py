"""The Hadamard-test device runs (pairs 3763, 3765), referee round 5 (2026-11-12): each block's calibration slope
against the slope its 256 overlaps show against their known ideal values, the error per circuit against shot noise, the
spread within and between rows, and the hybrid evidence with the calibration slope, the overlap slope, a per-row fit
(an oracle: fitted on the row's own readings) and no correction.

python device_errors.py   -> results/device_errors.json   (CPU, a few minutes)
"""
import json, math, re
import numpy as np, torch
from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

OUT = "results/device_errors.json"
tag = "esnli_Born-ap_b2_s0_e20"
RUNS = {3763: ["ae3d09c69431cc3d", "26df20031defaaa7"], 3765: ["a1d6be83e81a3267", "4327c6e5c51fae25"]}
model, d = load(tag)


def readings(keys):
    out = {}
    for b, k in enumerate(keys):
        rec = json.load(open(f"ibm_jobs/{k}/record.json"))
        counts = json.load(open(f"ibm_jobs/{k}/counts.json"))
        ov, cal = [], []
        for c, cnt in zip(rec["circuits"], counts):
            n = sum(cnt.values())
            v = (cnt.get("0", 0) - cnt.get("1", 0)) / n
            m = re.match(r"block (\d) head (\d) q(\d+) k(\d+) (Re|Im)$", c["note"])
            if m:
                ov.append((int(m[2]), int(m[3]), int(m[4]), m[5] == "Im", v, n)); continue
            m = re.match(r"block (\d) random calibration \d+ ideal ([-+.\d]+)$", c["note"])
            cal.append((float(m[2]), v, n))
        out[b] = (ov, cal)
    return out


class HwFrozen(Frozen):
    def __init__(self, rows):
        super().__init__(); self.rows, self.blk = rows, 0

    def __call__(self, x):
        if not self.replay and x.dim() == 4:
            x = x * self.rows[self.blk][None, :, :, None].to(x.dtype); self.blk += 1
        return super().__call__(x)


def hw_evidence(ids, target, rows, tok):
    x0, f = _score_fn(model, ids); x0 = x0.detach()
    fz = HwFrozen(rows)
    with torch.no_grad():
        f(x0, fz)
    total = torch.zeros_like(x0)
    for t, w in zip(*nodes01(100)):
        x = (float(t) * x0).requires_grad_(True)
        (g,) = torch.autograd.grad(f(x, fz.start_replay()).gather(1, target[:, None]).sum(), x)
        total += float(w) * g
    return (x0 * total).detach()[0, tok].numpy()


res = {}
for idx, keys in RUNS.items():
    ids = torch.tensor(d["x_test"][idx:idx + 1])
    tok = np.where(ids[0].numpy() != 0)[0]
    present = list(tok) + [ids.shape[1]]
    table = readings(keys)
    words = [d["words"][t] for t in ids[0][tok].tolist()]
    with torch.no_grad():
        pe = compiled_probs(model, ids)[0]
    target = pe.argmax(1)
    a = evidence(model, ids, target)[0].detach()[0, tok].numpy()
    blocks = []

    def make(mode):
        """mode: 'cal' (paper: slope+offset from calibrations), 'ovl' (slope+offset fitted on the overlaps
        themselves against their ideal values: an oracle block calibration), 'row' (one oracle slope per row
        (head, query): the best a per-row calibration could do), 'none' (no visibility correction)."""
        rows = []

        def overlap(qi, kj, block):
            q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
            H, L = q.shape[0], q.shape[1]
            ov, cal = table[block]
            x = np.array([c[0] for c in cal]); y = np.array([c[1] for c in cal])
            slope_c, off_c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
            exact = np.einsum("hie,hje->hij", q.conj(), k)
            ideal, meas, scl, hij = [], [], [], []
            for h, i, j, imag, v, n in ov:
                nn = np.linalg.norm(q[h, i]) * np.linalg.norm(k[h, j])
                ideal.append((exact[h, i, j].imag if imag else exact[h, i, j].real) / nn); meas.append(v); scl.append(nn); hij.append((h, i, j))
            ideal, meas = np.array(ideal), np.array(meas)
            slope_o, off_o = np.linalg.lstsq(np.vstack([ideal, np.ones_like(ideal)]).T, meas, rcond=None)[0]
            # per-row oracle slope through the origin after the calibration offset
            if mode == "cal":
                slope, off = slope_c, off_c
            elif mode == "ovl":
                slope, off = slope_o, off_o
            elif mode == "none":
                slope, off = 1.0, 0.0
            else:
                slope, off = slope_c, off_c
            re_, im_ = torch.zeros(1, H, L, L, dtype=torch.float64), torch.zeros(1, H, L, L, dtype=torch.float64)
            for (h, i, j, imag, v, n), nn in zip(ov, scl):
                (im_ if imag else re_)[0, h, i, j] = (v - off) * nn
            s_ex = torch.tensor(np.abs(exact) ** 2)
            sq = re_[0] ** 2 + im_[0] ** 2
            keep = torch.zeros(L, dtype=torch.bool); keep[present] = True
            if mode == "row":
                # per-row visibility fitted on that row's 2x(n+1) readings (oracle)
                eta2 = torch.ones(H, L, dtype=torch.float64)
                for h in range(H):
                    for i in present:
                        m = [(id_, me) for (hh, ii, jj), id_, me in zip(hij, ideal, meas - off) if hh == h and ii == i]
                        xi = np.array([t[0] for t in m]); yi = np.array([t[1] for t in m])
                        eta2[h, i] = (xi @ yi / (xi @ xi)) ** 2 if xi @ xi > 0 else 1.0
                s_hw = sq / eta2[:, :, None]
            else:
                s_hw = sq / slope ** 2
            num, den = (s_hw * keep).sum(-1), (s_ex * keep).sum(-1)
            rows.append(torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den)))
            if mode == "cal":
                # structure of the per-circuit error after the block calibration
                resid = (meas - off_c) / slope_c - ideal
                rowkey = np.array([h * 100 + i for h, i, j in hij])
                within = np.mean([resid[rowkey == rk].std() for rk in np.unique(rowkey)])
                rowmean = np.array([resid[rowkey == rk].mean() for rk in np.unique(rowkey)])
                shot_sd = float(np.sqrt((1 - np.clip(meas, -1, 1) ** 2).mean() / 2000) / slope_c)
                rr = rows[-1][:, present].numpy().ravel()
                blocks.append(dict(block=block, slope_cal=float(slope_c), off_cal=float(off_c), slope_overlaps=float(slope_o),
                                   off_overlaps=float(off_o), corr_meas_ideal=float(np.corrcoef(ideal, meas)[0, 1]),
                                   resid_sd_per_circuit=float(resid.std()), shot_sd_per_circuit=shot_sd,
                                   within_row_sd=float(within), between_row_sd_of_means=float(rowmean.std()),
                                   row_ratio_log_sd=float(np.log(rr).std()), row_ratio_log_mean=float(np.log(rr).mean()),
                                   cal_resid_sd=float(np.std(y - slope_c * x - off_c)),
                                   cal_ideal_abs_median=float(np.median(np.abs(x))), overlap_ideal_abs_median=float(np.median(np.abs(ideal)))))
            return re_, im_
        return overlap, rows

    r = dict(words=words, exact=a.tolist(), exact_p=pe[0].tolist())
    for mode in ("cal", "ovl", "row", "none"):
        overlap, rows = make(mode)
        with torch.no_grad():
            p = compiled_probs(model, ids, overlap=overlap)[0][0].numpy()
        b = hw_evidence(ids, target, rows, tok)
        r[mode] = dict(device_p=p.tolist(), corr=float(np.corrcoef(a, b)[0, 1]), l1_rel=float(np.abs(a - b).sum() / np.abs(a).sum()),
                       top_same=bool(np.argmax(a) == np.argmax(b)), evidence=b.tolist(),
                       row_log_sd=[float(np.log(rw[:, present].numpy().ravel()).std()) for rw in rows])
    r["blocks"] = blocks
    # uniform attention and random overlaps
    def uniform(qi, kj, block):
        H, L = qi.shape[1], qi.shape[2]
        re_ = torch.zeros(1, H, L, L, dtype=torch.float64)
        for i in present:
            for j in present:
                re_[0, :, i, j] = 1.0
        return re_, torch.zeros_like(re_)
    with torch.no_grad():
        r["uniform_p"] = compiled_probs(model, ids, overlap=uniform)[0][0].tolist()
    rng = np.random.default_rng(0)
    same = 0
    for _ in range(100):
        def rnd(qi, kj, block):
            H, L = qi.shape[1], qi.shape[2]
            re_ = torch.zeros(1, H, L, L, dtype=torch.float64); im_ = torch.zeros_like(re_)
            for i in present:
                for j in present:
                    re_[0, :, i, j] = torch.tensor(rng.standard_normal(H)); im_[0, :, i, j] = torch.tensor(rng.standard_normal(H))
            return re_, im_
        with torch.no_grad():
            same += int(compiled_probs(model, ids, overlap=rnd)[0][0].argmax() == target[0])
    r["random_overlaps_same_answer_of_100"] = same
    res[str(idx)] = r
    print(idx, json.dumps(r, indent=1), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
