"""How many shots the evidence needs (Proposition in Section 5): the hybrid evidence (exact numerators, measured
attention row sums D_i) under shot noise of compute-uncompute readings, against the bound.

Per test pair (Born Transformer seed 0):
  rows      every attention row i (block, head, slot) with D_i > 0; R of them
  A_i       sum_j n_ij^4 / D_i^2, n_ij = |q_i||k_j|, D_i = sum_j n_ij^2 r_ij, r_ij = |<k^_j|q^_i>|^2  (Hoeffding factor)
  G_i       d e / d log D_i at the exact row sums (central differences, step 1e-5; e is a polynomial in 1/D)
  gamma     sum_i |G_i|_1 / sum_t |e_t|            (row-sum sensitivity)
  rho       sum_t |e_t| / |sum_t e_t|              (cancellation)
  S_req     shots per circuit that the bound asks for evidence error eps |sum e| with probability 1 - delta:
            ln(2R/delta) max_i A_i (gamma rho)^2 / (2 eps^2)       (slope 1, offset 0)
Simulation: every reading p_ij ~ Binomial(S, r_ij) / S, D^_i = sum_j n_ij^2 p_ij, the evidence recomputed with D^;
  per S the share of draws with |e(D^) - e(D)|_1 <= eps |sum e|, and the first-order check: the actual error against
  the linear term sum_i (D^_i/D_i - 1) G_i and against the bound sum_i |D^_i/D_i - 1| |G_i|_1.

python shot_bound.py [--pairs 100] [--draws 40] [--eps 0.05] [--delta 0.05]   -> results/shot_bound.json  (GPU if any)
"""
import json
import math
import pathlib
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
NP, DRAWS, EPS, DELTA = int(opt("--pairs", 100)), int(opt("--draws", 40)), float(opt("--eps", 0.05)), float(opt("--delta", 0.05))
SHOTS = (500, 2000, 10000)
H_FD = 1e-5
TAG = "esnli_Born-ap_b2_s0_e20"
dev = "cuda" if torch.cuda.is_available() else "cpu"
model_cpu, d = load(TAG)
model, _ = load(TAG)
model = model.to(dev)
rng = np.random.default_rng(0)
pick = np.random.default_rng(1).choice(len(d["x_test"]), NP, replace=False)


class HwFrozen(Frozen):                 # attention row sums times rows[block] [B, H, L] (measured / exact)
    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, dd):
        if not self.replay and dd.dim() == 4:
            dd = dd * self.rows[self.blk][:, :, :, None].to(dd.dtype)
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


def capture(ids):
    cap = {}

    def ov(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
        cap[block] = (q, k)
        o = torch.tensor(np.einsum("hie,hje->hij", q.conj(), k))[None]
        return o.real.clone(), o.imag.clone()
    with torch.no_grad():
        compiled_probs(model_cpu, ids, overlap=ov)
    return cap


out = []
for n, idx in enumerate(pick):
    ids1 = torch.tensor(d["x_test"][idx:idx + 1])
    tok = np.where(ids1[0].numpy() != 0)[0]
    cap = capture(ids1)
    H, L = cap[0][0].shape[:2]
    nn2, rr, rows_idx = {}, {}, []
    for b in (0, 1):
        q, k = cap[b]
        nq, nk = np.linalg.norm(q, axis=-1), np.linalg.norm(k, axis=-1)          # [H, L]
        o2 = np.abs(np.einsum("hie,hje->hij", q.conj(), k)) ** 2
        n2 = (nq[:, :, None] * nk[:, None, :]) ** 2
        r = np.divide(o2, n2, out=np.zeros_like(o2), where=n2 > 0)
        nn2[b], rr[b] = n2, r
        for h in range(H):
            for i in range(L):
                if (n2[h, i] * r[h, i]).sum() > 0:
                    rows_idx.append((b, h, i))
    R = len(rows_idx)
    D = {key: float((nn2[key[0]][key[1], key[2]] * rr[key[0]][key[1], key[2]]).sum()) for key in rows_idx}
    A = {key: float((nn2[key[0]][key[1], key[2]] ** 2).sum()) / D[key] ** 2 for key in rows_idx}
    V = {key: float((nn2[key[0]][key[1], key[2]] ** 2 * rr[key[0]][key[1], key[2]]
                     * (1 - rr[key[0]][key[1], key[2]])).sum()) / D[key] ** 2 for key in rows_idx}
    ids = ids1.to(dev)
    with torch.no_grad():
        target = model(ids).argmax(1)

    def ev_batch(factors):                   # factors: [B, R] row factors D^/D -> evidence [B, n_tok]
        B = factors.shape[0]
        rows = [torch.ones(B, H, L, dtype=torch.float64, device=dev) for _ in (0, 1)]
        for c, (b, h, i) in enumerate(rows_idx):
            rows[b][:, h, i] = factors[:, c]
        out_ = []
        for s in range(0, B, 64):
            sl = slice(s, s + 64)
            bb = rows[0][sl].shape[0]
            out_.append(hw_ev(ids.expand(bb, -1), target.expand(bb), [rows[0][sl], rows[1][sl]])[:, tok])
        return torch.cat(out_).cpu().numpy()

    ex = evidence(model, ids, target)[0][0, tok].detach().cpu().numpy()
    one = torch.ones(1, R, dtype=torch.float64, device=dev)
    e1 = ev_batch(one)[0]
    fd = torch.ones(2 * R, R, dtype=torch.float64, device=dev)
    for c in range(R):
        fd[2 * c, c], fd[2 * c + 1, c] = math.exp(H_FD), math.exp(-H_FD)
    ef = ev_batch(fd)
    G = (ef[0::2] - ef[1::2]) / (2 * H_FD)                                        # [R, n_tok]
    l1 = np.abs(ex).sum()
    tot = abs(ex.sum())
    gamma, rho = float(np.abs(G).sum() / l1), float(l1 / tot)
    Amax = max(A.values())
    g1 = np.abs(G).sum(1)                                                          # |G_i|_1
    g2 = (G ** 2).sum(1)                                                           # |G_i|_2^2
    Av, Vv = np.array([A[key] for key in rows_idx]), np.array([V[key] for key in rows_idx])
    c_a = math.sqrt(math.log(2 * R / DELTA) / 2) * float((g1 * np.sqrt(Av)).sum()) / tot   # bound_a(S) = c_a / sqrt(S)
    c_b = math.sqrt(float((Vv * g2).sum())) / tot                                         # pred_b(S) = c_b / sqrt(S)
    rec = dict(index=int(idx), n_tok=int(len(tok)), R=R, rho=rho, gamma=gamma, A_max=Amax,
               A_median=float(np.median(Av)), c_a=c_a, c_b=c_b, S_req_a=(c_a / EPS) ** 2, S_req_b=(c_b / EPS) ** 2,
               S_req_uniform=math.log(2 * R / DELTA) * Amax * (gamma * rho) ** 2 / (2 * EPS ** 2),
               check_ev_vs_exact=float(np.abs(e1 - ex).max()), sim={})
    for S in SHOTS:
        fac = np.empty((DRAWS, R))
        for c, (b, h, i) in enumerate(rows_idx):
            n2, r = nn2[b][h, i], rr[b][h, i]
            m = n2 > 0
            p = rng.binomial(S, np.clip(r[m], 0, 1), size=(DRAWS, int(m.sum()))) / S
            fac[:, c] = (p * n2[m]).sum(1) / D[(b, h, i)]
        es = ev_batch(torch.tensor(fac, device=dev))
        err = np.abs(es - ex).sum(1)
        lin = np.abs(((fac - 1) @ G) - (es - ex)).sum(1)                            # remainder of the linear term
        bnd = np.abs(fac - 1) @ np.abs(G).sum(1)                                     # first-order bound
        l2 = np.sqrt(((es - ex) ** 2).sum(1))
        rec["sim"][str(S)] = dict(share_within=float((err <= EPS * tot).mean()), err_median=float(np.median(err) / tot),
                                  err_p95=float(np.percentile(err, 95) / tot), bound_a=c_a / math.sqrt(S),
                                  l2_rms=float(np.sqrt((l2 ** 2).mean()) / tot), pred_b=c_b / math.sqrt(S),
                                  remainder_over_err_max=float((lin / np.maximum(err, 1e-300)).max()),
                                  err_over_bound_max=float((err / bnd).max()),
                                  row_rel_err_max=float(np.abs(fac - 1).max()))
    out.append(rec)
    print(n, json.dumps(rec), flush=True)

summ = {k: dict(median=float(np.median([r[k] for r in out])), p90=float(np.percentile([r[k] for r in out], 90)),
                max=float(max(r[k] for r in out))) for k in ("rho", "gamma", "A_max", "S_req_a", "S_req_b", "S_req_uniform", "n_tok", "R")}
summ["share_within"] = {S: float(np.mean([r["sim"][str(S)]["share_within"] for r in out])) for S in map(str, SHOTS)}
summ["bound_a_holds"] = {S: float(np.mean([r["sim"][S]["err_p95"] <= r["sim"][S]["bound_a"] for r in out])) for S in map(str, SHOTS)}
summ["bound_a_over_p95"] = {S: float(np.median([r["sim"][S]["bound_a"] / r["sim"][S]["err_p95"] for r in out])) for S in map(str, SHOTS)}
summ["l2_over_pred_b"] = {S: dict(median=float(np.median([r["sim"][S]["l2_rms"] / r["sim"][S]["pred_b"] for r in out])),
                                  low=float(min(r["sim"][S]["l2_rms"] / r["sim"][S]["pred_b"] for r in out)),
                                  high=float(max(r["sim"][S]["l2_rms"] / r["sim"][S]["pred_b"] for r in out))) for S in map(str, SHOTS)}
summ["corr_rho_S_req_b"] = float(np.corrcoef(np.log([r["rho"] for r in out]), np.log([r["S_req_b"] for r in out]))[0, 1])
summ["err_over_bound_max"] = max(r["sim"][s]["err_over_bound_max"] for r in out for s in map(str, SHOTS))
summ["remainder_over_err_max"] = {S: max(r["sim"][S]["remainder_over_err_max"] for r in out) for S in map(str, SHOTS)}
summ["check_ev_vs_exact_max"] = max(r["check_ev_vs_exact"] for r in out)
print(json.dumps(summ, indent=1))
json.dump(dict(tag=TAG, eps=EPS, delta=DELTA, draws=DRAWS, shots=SHOTS, summary=summ, pairs=out),
          open("results/shot_bound.json", "w"), indent=1)
