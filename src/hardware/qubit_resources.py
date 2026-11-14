"""Quantum resources and errors of the qubit realisations (referee round 2, for Quantum). Simulated, no device.

Born Transformer (seed 0), attention overlaps read by Hadamard tests, everything else exact, as on ibm_kingston:
  C1  shot noise: each <q_i|k_j> estimated from S shots per circuit (Re and Im, P(0) - P(1) of normalised states,
      then times |q_i||k_j|); answer agreement with the exact model and |dP| against S; how the power in the weight and the
      row normalisation pass the overlap error on to the weights;
  C2  the hybrid evidence: exact numerators, attention row sums from the noisy overlaps; its correlation and top-3
      agreement with the exact evidence against S, and the cancellation ratio sum_t|e_t| / |sum_t e_t| that makes it
      fragile (a relative error of the row sums is multiplied by about that ratio in the evidence);
  C3  visibility: a loss eta common to a row cancels in w_ij; a loss that differs from circuit to circuit (eta_ij =
      eta0 (1 + sigma z)) does not - answer agreement against sigma at 10^4 shots;
  C5  shots per sentence: 8 (n + 1)^2 circuits x the shots per circuit that keep 99 % of answers, against n.
Coherent attention layer (all maps brickwall):
  C4  post-selection probability in closed form, |y|^2 / (|x_bar|^2 (sum_j |x_j|^2)^2 2^5 2^p) (checked against the
      circuit's statevector, 2.42e-5 and 1.85e-4 on pairs 3765 and 3763), over the test pairs against the number of
      slots; repetitions to one accepted shot, 1/P, and with amplitude amplification, about pi / (4 sqrt P) rounds of
      two passes each.

python qubit_resources.py [--n 300] [--tag esnli_Born-ap_b2_s0_e20]   -> results/qubit_resources_<tag>.json
"""
import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path[:0] = sorted(str(p) for p in pathlib.Path(__file__).resolve().parents[1].iterdir()
                      if p.is_dir() and p.name != "__pycache__")       # the folders of src/: modules import by name
import synth
from evidence import _score_fn, nodes01
from models import Frozen, build

dev = "cuda" if torch.cuda.is_available() else "cpu"
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)[:N]
V, C = len(d["words"]), len(d["classes"])
gen = torch.Generator(device=dev).manual_seed(0)


def load(tag):
    cfg = json.load(open(f"results/{tag}.json"))
    m = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=64).to(dev)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
    return m.double().eval()


TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "esnli_Born-ap_b2_s0_e20"
model = load(TAG)
POWER = 2 if model.attn == "born2" else 1               # weight = |<q|k>|^(2 POWER)
NODES = math.ceil(model.degree() / 2)                    # exact Gauss-Legendre nodes, as evidence.py

NOISE = {"S": None, "sigma": 0.0}                     # read by the einsum hook while set


def noisy_overlap(ov, q, k):
    """Hadamard-test estimate of ov = <q_i|k_j> [B, H, L, L]: normalised Re and Im each from S shots, times |q||k|;
    a visibility eta_ij = 1 + sigma z per circuit (uniform loss cancels in w_ij and is left out)."""
    nq = torch.linalg.vector_norm(q, dim=-1).permute(0, 2, 1)[..., None]        # [B, H, L, 1]
    nk = torch.linalg.vector_norm(k, dim=-1).permute(0, 2, 1)[..., None, :]     # [B, H, 1, L]
    scale = (nq * nk).clamp(min=1e-300)
    r = ov / scale

    def est(x):
        if NOISE["sigma"]:
            x = x * (1 + NOISE["sigma"] * torch.randn(x.shape, generator=gen, device=dev, dtype=x.dtype))
        if NOISE["S"] is None:
            return x
        p0 = ((1 + x.clamp(-1, 1)) / 2)
        cnt = torch.distributions.Binomial(NOISE["S"], probs=p0).sample()
        return 2 * cnt / NOISE["S"] - 1
    return torch.complex(est(r.real), est(r.imag)) * scale


orig_einsum = torch.einsum


def hooked(eq, *ops):
    out = orig_einsum(eq, *ops)
    if eq == "bihe,bjhe->bhij" and (NOISE["S"] is not None or NOISE["sigma"]):
        q, k = ops
        out = noisy_overlap(out, q.conj(), k)
    return out


torch.einsum = hooked


def probs(ids):
    with torch.no_grad():
        return model(ids)                                                      # P_g, the Born probabilities


def run_noisy(ids, S, sigma=0.0):
    NOISE.update(S=S, sigma=sigma)
    try:
        return probs(ids)
    finally:
        NOISE.update(S=None, sigma=0.0)


def frozen(ids, S=None):
    """The denominators of one pass (noisy overlaps if S), as models.Frozen records them."""
    x0, f = _score_fn(model, ids)
    fz = Frozen()
    NOISE.update(S=S, sigma=0.0)
    try:
        with torch.no_grad():
            f(x0, fz)
    finally:
        NOISE.update(S=None, sigma=0.0)
    return fz


def evidence_with(ids, target, fz, n_nodes=None):
    x0, f = _score_fn(model, ids)
    total = torch.zeros_like(x0)
    for t, w in zip(*nodes01(n_nodes or NODES)):
        x = (float(t) * x0).requires_grad_(True)
        out = f(x, fz.start_replay()).gather(1, target[:, None]).sum()
        (g,) = torch.autograd.grad(out, x)
        total += float(w) * g
    return (x0 * total).detach()


ids_all = torch.tensor(d["x_test"][pick]).to(dev)
res = {"tag": TAG, "pairs": N, "nodes": NODES}

# ---- exact pass
with torch.no_grad():
    P_ex = torch.cat([probs(ids_all[i:i + 50]) for i in range(0, N, 50)])
ans_ex = P_ex.argmax(1)

# ---- C1 shots
res["C1"] = {}
for S in (100, 300, 1000, 2000, 10000, 100000):
    P = torch.cat([run_noisy(ids_all[i:i + 50], S) for i in range(0, N, 50)])
    dP = (P - P_ex).abs().max(1).values
    res["C1"][S] = dict(agree=float((P.argmax(1) == ans_ex).double().mean()), dP_median=float(dP.median()),
                        dP_p95=float(dP.quantile(0.95)))
    print("C1", S, res["C1"][S], flush=True)

# error passed from an overlap to a weight: relative error of |<q|k>|^(2 POWER) vs of |<q|k>| at 2,000 shots
cap = {}


def grab(eq, *ops):
    out = orig_einsum(eq, *ops)
    if eq == "bihe,bjhe->bhij":
        cap.setdefault("ov", []).append((out, ops[0].conj(), ops[1]))
    return out


torch.einsum = grab
with torch.no_grad():
    model(ids_all[:50])
torch.einsum = hooked
ov, q, k = cap["ov"][0]
NOISE.update(S=2000)
ovn = noisy_overlap(ov, q, k)
NOISE.update(S=None)
mag, magn = ov.abs(), ovn.abs()
big = mag > mag.flatten().quantile(0.5)                                   # the overlaps that carry the weight
rel1 = ((magn - mag).abs() / mag)[big].median()
e = 2 * POWER
relw = ((magn ** e - mag ** e).abs() / mag ** e)[big].median()
res["C1_amplification"] = dict(shots=2000, exponent=e, rel_err_overlap=float(rel1), rel_err_weight=float(relw),
                               ratio=float(relw / rel1))
print("C1 amplification", res["C1_amplification"], flush=True)

# ---- C2 hybrid evidence
res["C2"] = {}
n2 = min(N, 100)
ids2 = ids_all[:n2]
ex_ev, kappa = [], []
for i in range(n2):
    ids = ids2[i:i + 1]
    t = ans_ex[i:i + 1]
    fz = frozen(ids)
    e = evidence_with(ids, t, fz)[0]
    pres = ids[0] != 0
    e = e[:pres.shape[0]][pres]
    ex_ev.append(e)
    kappa.append(float(e.abs().sum() / e.sum().abs().clamp(min=1e-300)))
res["C2_cancellation"] = dict(median=float(np.median(kappa)), p90=float(np.quantile(kappa, 0.9)))
for S in (2000, 10000, 100000, 1000000):
    corr, top = [], []
    for i in range(n2):
        ids = ids2[i:i + 1]
        t = ans_ex[i:i + 1]
        fe, fn = frozen(ids), frozen(ids, S)
        hy = Frozen()
        hy.values = [vn if ve.dim() == 4 else ve for ve, vn in zip(fe.values, fn.values)]   # attention row sums
        e = evidence_with(ids, t, hy)[0]
        pres = ids[0] != 0
        e = e[:pres.shape[0]][pres]
        a, b = ex_ev[i].cpu().numpy(), e.cpu().numpy()
        corr.append(float(np.corrcoef(a, b)[0, 1]))
        kk = min(3, len(a))
        top.append(len(set(np.argsort(-a)[:kk]) & set(np.argsort(-b)[:kk])) / kk)
    res["C2"][S] = dict(corr_median=float(np.median(corr)), top3=float(np.mean(top)))
    print("C2", S, res["C2"][S], flush=True)
# row sums are held denominators: unlike the weights, they feel the visibility. A calibration of each row's eta off
# by a relative sigma scales its row sum by (1 + sigma z)^(2 POWER) (here with 10^4 shots on top)
res["C2_calibration"] = {}
for sig in (0.01, 0.03, 0.1, 0.3):
    corr, top = [], []
    for i in range(n2):
        ids = ids2[i:i + 1]
        t = ans_ex[i:i + 1]
        fe, fn = frozen(ids), frozen(ids, 10000)
        hy = Frozen()
        hy.values = [vn * (1 + sig * torch.randn(vn.shape, generator=gen, device=dev, dtype=vn.dtype)) ** (2 * POWER)
                     if ve.dim() == 4 else ve for ve, vn in zip(fe.values, fn.values)]
        e = evidence_with(ids, t, hy)[0]
        pres = ids[0] != 0
        e = e[:pres.shape[0]][pres]
        a, b = ex_ev[i].cpu().numpy(), e.cpu().numpy()
        corr.append(float(np.corrcoef(a, b)[0, 1]))
        kk = min(3, len(a))
        top.append(len(set(np.argsort(-a)[:kk]) & set(np.argsort(-b)[:kk])) / kk)
    res["C2_calibration"][sig] = dict(corr_median=float(np.median(corr)), top3=float(np.mean(top)))
    print("C2 calibration", sig, res["C2_calibration"][sig], flush=True)
for idx in (3763, 3765):                                   # the device pairs
    ids = torch.tensor(d["x_test"][idx:idx + 1]).to(dev)
    with torch.no_grad():
        t = probs(ids).argmax(1)
    e = evidence_with(ids, t, frozen(ids))[0][:ids.shape[1]][ids[0] != 0]
    res["C2_cancellation"][f"pair {idx}"] = float(e.abs().sum() / e.sum().abs())
print("C2 cancellation", res["C2_cancellation"], flush=True)

# ---- C3 visibility differing per circuit
res["C3"] = {}
for sigma in (0.0, 0.02, 0.05, 0.1, 0.2):
    P = torch.cat([run_noisy(ids_all[i:i + 50], 10000, sigma) for i in range(0, N, 50)])
    res["C3"][sigma] = dict(agree=float((P.argmax(1) == ans_ex).double().mean()),
                            dP_median=float((P - P_ex).abs().max(1).values.median()))
    print("C3", sigma, res["C3"][sigma], flush=True)

# ---- C5 shots per sentence
n_tok = (ids_all != 0).sum(1).cpu().numpy()
S99 = min([S for S, r in res["C1"].items() if r["agree"] >= 0.99], default=None)
res["C5"] = dict(shots_per_circuit_99=S99, lengths={})
for n in (7, 15, 23, 40, 64):
    circ = 8 * (n + 1) ** 2
    res["C5"]["lengths"][n] = dict(circuits=circ, shots=circ * S99 if S99 else None)
res["C5"]["median_length"] = int(np.median(n_tok))
print("C5", res["C5"], flush=True)

# ---- C4 coherent layer pass probability
torch.einsum = orig_einsum
cm = load("esnli_Coherent-attn-b12-24_b1_s0")
rows = []
for i in pick[:N]:
    ids = torch.tensor(d["x_test"][i:i + 1]).to(dev)
    ids = ids[:, :int((ids != 0).sum())]
    with torch.no_grad():
        amp = cm.amplitudes(ids)
        L = ids.shape[1]
        seg = (torch.cumsum((ids == cm.sep).long(), 1) > 0).long()
        e = cm._c(cm.emb)[ids] + cm._c(cm.pos)[:L] + cm._c(cm.seg)[seg]
        tt = torch.arange(L, device=dev, dtype=cm.omega.dtype)
        e = e * torch.polar(torch.ones_like(tt[:, None] * cm.omega), tt[:, None] * cm.omega)
        x = e * amp.unsqueeze(-1).to(e.dtype)
        x = torch.cat((x, cm._c(cm.ref).expand(1, 1, -1)), 1)
        y = cm.values(ids, amp)
    slots = x.shape[1]
    p = math.ceil(math.log2(slots))
    P = float((y.abs() ** 2).sum() / ((x.sum(1).abs() ** 2).sum() * ((x.abs() ** 2).sum()) ** 2 * 2 ** 5 * 2 ** p))
    rows.append((slots, p, P))
rows = np.array(rows)
res["C4"] = {}
for p in sorted(set(rows[:, 1].astype(int))):
    sel = rows[rows[:, 1] == p]
    Pm = float(np.median(sel[:, 2]))
    res["C4"][int(p)] = dict(pairs=int(len(sel)), slots=f"{int(sel[:, 0].min())}-{int(sel[:, 0].max())}",
                             qubits=17 + 2 * int(p), pass_median=Pm, pass_min=float(sel[:, 2].min()),
                             repetitions=1 / Pm, amplification_rounds=math.pi / (4 * math.sqrt(Pm)))
    print("C4", p, res["C4"][int(p)], flush=True)
lg = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 2]), 1)
res["C4_fit"] = dict(exponent=float(lg[0]), note="pass ~ slots^exponent over the test pairs")
print("C4 fit", res["C4_fit"])
json.dump(res, open(f"results/qubit_resources_{TAG.split('/')[-1]}.json", "w"), indent=1)
