"""Compile a trained (unconstrained) Born model to passive optics, exactly, and simulate the bench's noise.

Every step of a Born block is linear in the state except the MLP's product, which is measured; attention weights
and the Born readout do not change when the state is scaled. So a trained model needs no gain inside: scale each
matrix down to a contraction, attenuate the branches with passive splitters and combiners, and put the lost scale
back as a classical gain on the measured MLP numbers. What it costs is light (README, "Plan: hardware").

Light count, as amplitude factors on the state (source power 1 = the token fields, each of norm <= 1):
- attention: h is split four ways (Q, K, V, residual: 1/2 each); each q_i, k_j, v_j fans out to the n slots
  present (1/sqrt n); Re and Im of <q_i|k_j> take one more split (1/sqrt 2). The weighted sum y_i = sum_j w_ij v_j
  is passive (sum_j w_ij^2 <= 1). The residual combiner: alpha h / 2 + beta O y / (2 sqrt n), |alpha|^2 +
  |beta|^2 <= 1, which keeps 1 / (2 sqrt(1 + g^2 n)) of h + g O y, g = sigma_V sigma_O.
- MLP: h is split three ways (W, V, residual: 1/sqrt 3); Re[(W h)_c conj(V h)_c] is a balanced interferometric reading.
  The update U (gain * mid) is fresh light from the modulator (fields <= 1), so the combiner keeps
  alpha / sqrt 3 of h + update with alpha^2 (1 + u^2 / 3) <= 1, u the largest update relative to it.
- readout: the token fields add coherently in one combiner (1/sqrt n), then the camera.

Noise (optional): Poisson photon counts at every measurement for `photons` source photons per sentence - balanced
pairs for <q|k> (Re, Im) and the MLP's balanced reading, plain counts at the readout camera; and modulator precision, `bits`
per real part of every encoded field (token fields, attention weights, MLP update) and `wbits` per entry of every
fixed matrix (each relative to its largest value).

    python compile_hw.py <tag> [<tag> ...]                      # P_g of the compiled pass against the trained model
    python compile_hw.py --noise <tag> [<tag> ...]              # accuracy against photons per sentence and bits
    python compile_hw.py --noise --qubit <tag> [<tag> ...]      # accuracy against shots per estimate; rough cost
"""
import json
import math
import sys

import torch

import models
import synth


def sig(m):
    """Largest singular value, at least 1: what a matrix must be divided by to be a contraction."""
    return torch.linalg.matrix_norm(m, ord=2).max().clamp(min=1)


def vmax(x, amp):
    """Largest token field, a_t taken out (amplitudes are equal over the tokens present)."""
    return torch.linalg.vector_norm(x, dim=-1).max() / amp.max()


def quant(x, bits):
    """Uniform quantisation of each real part to `bits` bits (sign included), relative to the largest value."""
    if bits is None:
        return x
    if x.is_complex():
        return torch.complex(quant(x.real, bits), quant(x.imag, bits))
    top = x.abs().max().clamp(min=1e-300)
    step = top / (2 ** (bits - 1) - 1)
    return torch.round(x / step) * step


def counts(rate):
    """Poisson counts; above 1e4 the normal approximation (a double cannot hold counts beyond ~1e16 exactly)."""
    big = rate > 1e4
    normal = rate + rate.clamp(min=0).sqrt() * torch.randn_like(rate)
    return torch.where(big, normal, torch.poisson(torch.where(big, torch.zeros_like(rate), rate)))


def balanced(x, y, f, photons, dims):
    """Re<x|y> read by a balanced pair: counts of |x f + y f|^2 / 2 and |x f - y f|^2 / 2 (summed over `dims`)
    for `photons` source photons; without photons, the exact value."""
    exact = (x.conj() * y).real.sum(dims) if dims else (x.conj() * y).real
    if photons is None:
        return exact
    def inten(z):
        z2 = z.real ** 2 + z.imag ** 2 if z.is_complex() else z ** 2
        return z2.sum(dims) if dims else z2
    rp, rm = photons * f * f * inten(x + y) / 2, photons * f * f * inten(x - y) / 2
    # the difference of the two counts, drawn as (exact difference + noise) where both are large, so that two
    # nearly equal huge counts do not cancel in floating point
    big = (rp + rm) > 1e4                                                 # Skellam -> normal
    diff_big = 2 * photons * f * f * exact + (rp + rm).sqrt() * torch.randn_like(rp)
    small = torch.where(big, torch.zeros_like(rp), rp), torch.where(big, torch.zeros_like(rm), rm)
    diff = torch.where(big, diff_big, torch.poisson(small[0]) - torch.poisson(small[1]))
    return diff / (2 * photons * f * f)


def compiled_probs(model, ids, photons=None, bits=None, wbits=None, qubit=False, trace=None, overlap=None):
    """P_g by the compiled pass: contractions only, token fields of norm <= 1, passive splitters and combiners,
    classical gains on measured numbers; returns (P_g, log10 of the share of source light at the readout camera). Without
    noise it equals the trained model's P_g and is differentiable: `train.py --power` uses the second value.

    qubit=True: the qubit profile. No copies of a state: every reading prepares its states afresh, so there is no
    split or fan-out; each slot is prepared alone (its own field, a_t taken out) and the combiners are LCUs, which
    keep 1 / (1-norm of the coefficients): 1 / (1 + g) at attention, 1 / (1 + u) at the MLP, 1 / (sum_t a_t + 1)
    for the coherent readout (the second value is then log10 of the readout's post-selection probability). `photons` is then the
    shots per estimate: one Hadamard test per <q_i|k_j> (Re, Im), one per token for the MLP (all channels at once:
    ancilla and channel register measured together), and the readout. Counts are Poisson-approximated.

    trace: a list that receives (stage, value) pairs on the way (the fields, attention weights, MLP readings and
    camera intensities), for figures.

    overlap: a callable (q, k, block) -> (Re, Im) of <q_i|k_j>, [B, H, L, L], in place of the balanced readings
    (q [B, H, L, 1, e], k [B, H, 1, L, e]); `hw_attention.py` reads them from simulated quantum circuits."""
    log = (lambda name, v: trace.append((name, v.detach()))) if trace is not None else (lambda name, v: None)
    amp = model.amplitudes(ids)
    n = ((ids != 0).sum(1) + (1 if model.mean_channel else 0)).to(amp.dtype)[:, None, None]   # slots, [B, 1, 1]
    h = model.encode(ids, amp, lambda d: d)
    c0 = vmax(h, amp)
    h = quant(h / c0, bits)                                                # every token field <= 1 now
    scale = 1 / c0                                                         # h_compiled = scale * h_trained
    log("fields", h)
    dh = model.d // model.heads
    cx = model.cx
    # each slot's own amplitude a_t (1 for the reference slot and absent slots), to prepare a slot alone
    alpha = torch.where(amp > 0, amp, torch.ones_like(amp))
    if model.mean_channel:
        alpha = torch.cat((alpha, torch.ones_like(alpha[:, :1])), 1)
    alpha = alpha.to(h.real.dtype if cx else h.dtype)
    for block, (qkvo, wv, u) in enumerate(zip(model.qkvo, model.wv, model.u)):
        m = model._c(qkvo)
        cm = [quant(mi / sig(mi), wbits) for mi in m]
        q, k, v = (torch.einsum("btd,ed->bte", h, cm[i]).unflatten(-1, (model.heads, dh)) for i in range(3))
        # <q_i|k_j> per head: fields q_i, k_j at 1/2 (split) * 1/sqrt n (fan-out) * 1/sqrt 2 (Re, Im)
        f = (1 / (2 * torch.sqrt(n) * math.sqrt(2)))[:, :, :, None]          # [B, 1, 1, 1]
        qi, kj = q.permute(0, 2, 1, 3)[:, :, :, None], k.permute(0, 2, 1, 3)[:, :, None]   # [B,H,L,1,e], [B,H,1,L,e]
        pair = 1.0
        if qubit:                                                          # Hadamard test: P(+) - P(-) = Re<x|y>
            f, ai, aj = 1 / math.sqrt(2), alpha[:, None, :, None, None], alpha[:, None, None, :, None]
            qi, kj, pair = qi / ai, kj / aj, (ai * aj)[..., 0]
        if overlap is not None:
            re, im = overlap(qi, kj, block)
        else:
            re = balanced(qi, kj, f, photons, -1) * pair
            im = balanced(qi, kj * 1j, f, photons, -1) * pair if cx else torch.zeros_like(re)
        s = re ** 2 + im ** 2                                              # |<q_i|k_j>|^2 (Re<q|ik> = Im<q|k>)
        if model.attn == "born2":
            s = s ** 2
        w = s / (s.sum(-1, keepdim=True) + 1e-300)
        mask = (ids == 0)
        if model.mean_channel:
            mask = torch.cat((mask, torch.zeros_like(mask[:, :1])), 1)
        # no slot, no light: an absent token neither sends (column) nor receives (row); without the row mask, noise
        # on an all-zero row would be normalised into weights and light an empty slot
        w = w.masked_fill(mask[:, None, None, :] | mask[:, None, :, None], 0)
        w = quant(w, bits)                                                 # programmed attenuators
        log("attention", w)
        y = torch.einsum("bhij,bjhe->bihe", w.to(v.dtype), v).flatten(-2) @ cm[3].T
        g = sig(m[2]) * sig(m[3])                                          # what V, O lost
        keep = 1 / (1 + g) if qubit else 1 / (2 * torch.sqrt(1 + g * g * n))
        h, scale = keep * (h + g * y), scale * keep
        log("after attention", h)
        w2 = model._c(wv)
        cw = [quant(wi / sig(wi), wbits) for wi in w2]
        a, b = h @ cw[0].T, h @ cw[1].T
        if qubit:
            at = alpha[:, :, None]
            mid = balanced(a / at, b / at, 1 / math.sqrt(2), photons, None) * at * at
        else:
            mid = balanced(a, b, 1 / math.sqrt(3), photons, None)        # Re[conj(a) b] = Re[a conj b], per channel
        uu = model._c(u)
        cu = quant(uu / sig(uu), wbits)
        gw = sig(w2[0]) * sig(w2[1])
        gain = gw / scale * sig(uu)                                        # classical, on measured numbers
        upd = quant((mid * gain).to(h.dtype) @ cu.T, bits)                # fresh light from the modulator
        big = vmax(upd, amp)                                               # the modulator's field must be <= 1
        keep = 1 / (1 + big) if qubit else 1 / torch.sqrt(3 + big * big)  # LCU; or alpha / sqrt 3 with
        # alpha^2 (1 + big^2 / 3) <= 1
        log("mlp reading", mid)
        h, scale = keep * (h + upd), scale * keep
        log("after mlp", h)
    rr = model._c(model.readout)
    r = h @ quant(rr / sig(rr), wbits).T
    if qubit:
        # LCU over the slots: coefficient a_t on each token's own state r_t / a_t (1 on the reference), so the
        # sum is the model's; it keeps 1 / (sum_t a_t + 1)
        norm1 = amp.sum(1, keepdim=True) + (1 if model.mean_channel else 0)
        phi = r.sum(1) / norm1.to(alpha.dtype)
    else:
        phi = r.sum(1) / torch.sqrt(n[:, 0])                               # one combiner over the slots
    inten = (phi.real ** 2 + phi.imag ** 2) if cx else phi ** 2            # share of the source light, per mode
    power = torch.log10(inten.sum(-1).mean())
    if photons is not None:
        inten = counts(photons * inten) / photons
    log("camera", inten)
    p = inten @ model.groups.T / inten.sum(-1, keepdim=True).clamp(min=1e-300)
    return p, power


def qubit_cost(model, ids):
    """Rough cost of one sentence on qubits: estimates (Hadamard tests) per sentence, qubits, and two-qubit gates to
    prepare the readout state. Gate model, deliberately rough: preparing a d-vector ~ d gates, a dense contraction
    (a unitary on 2d dims) ~ (2d)^2; no copies, so a state after a block is prepared by re-running the block, and an
    LCU over n slots prepares n states: P' = P + n (P + C) + C + d per block."""
    n = float(((ids != 0).sum(1) + (1 if model.mean_channel else 0)).double().mean())
    d, dff = model.d, model.u[0].shape[-1]
    per_block = model.heads * n * n * (2 if model.cx else 1) + n
    estimates = model.n_blocks * per_block + 1
    c = (2 * max(d, dff)) ** 2
    p = d
    for _ in range(model.n_blocks):
        p = p + n * (p + c) + c + d
    qubits = math.ceil(math.log2(n)) + math.ceil(math.log2(2 * max(d, dff))) + 1 + model.n_blocks * 2
    return dict(slots=n, estimates=estimates, qubits=qubits, gates_readout=p + c)


def load(tag):
    cfg = json.load(open(f"results/{tag}.json"))
    d = synth.get(cfg.get("data", "banking77"))
    model = models.build(cfg["model"], len(d["words"]), len(d["classes"]), blocks=cfg["blocks"],
                         max_len=d["x_test"].shape[1]).double()
    model.load_state_dict({k: v.double() for k, v in torch.load(f"results/{tag}.pt", map_location="cpu").items()})
    return model.eval(), d


if __name__ == "__main__":
    noise = "--noise" in sys.argv
    qubit = "--qubit" in sys.argv
    for tag in [a for a in sys.argv[1:] if not a.startswith("--")]:
        model, d = load(tag)
        ids, y = torch.tensor(d["x_test"][:512]), torch.tensor(d["y_test"][:512])
        torch.manual_seed(0)
        with torch.no_grad():
            ref = model(ids)
            p, power = compiled_probs(model, ids, qubit=qubit)
            acc = float((p.argmax(1) == y).double().mean())
            print(f"{tag}: max |P_compiled - P_trained| {float((p - ref).abs().max()):.1e}, accuracy {acc:.3f}, " +
                  (f"readout post-selection 10^{float(power):.1f}" if qubit else
                   f"light at the camera 10^{float(power):.1f}"), flush=True)
            if not noise:
                continue
            res = dict(tag=tag, noiseless=acc, light_log10=float(power), photons={}, bits={})
            if qubit:
                cost = qubit_cost(model, ids)
                res = dict(res, profile="qubit", cost=cost, shots={})
                print(f"  slots {cost['slots']:.1f}, estimates/sentence {cost['estimates']:.0f}, qubits ~{cost['qubits']}, "
                      f"two-qubit gates to the readout ~{cost['gates_readout']:.1e} (rough)")
                for lg in range(2, 12):
                    a = float((compiled_probs(model, ids, photons=10.0 ** lg, qubit=True)[0].argmax(1) == y)
                              .double().mean())
                    res["shots"][lg] = a
                print("  shots/estimate  " + "  ".join(f"1e{k}:{v:.3f}" for k, v in res["shots"].items()))
                json.dump(res, open(f"results/{tag}_qubit.json", "w"), indent=1)
                continue
            for lg in range(4, 22, 2):
                a = float((compiled_probs(model, ids, photons=10.0 ** lg)[0].argmax(1) == y).double().mean())
                res["photons"][lg] = a
            for b in (4, 6, 8, 10, 12):
                a = float((compiled_probs(model, ids, bits=b, wbits=b)[0].argmax(1) == y).double().mean())
                res["bits"][b] = a
            print("  photons/sentence  " + "  ".join(f"1e{k}:{v:.3f}" for k, v in res["photons"].items()))
            print("  bits (fields = matrices)  " + "  ".join(f"{k}:{v:.3f}" for k, v in res["bits"].items()))
            json.dump(res, open(f"results/{tag}_noise.json", "w"), indent=1)
