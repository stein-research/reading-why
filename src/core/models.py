"""Three model families of the same budget, all taking padded token ids (0 = absent).

S   softmax transformer: pre-LayerNorm, softmax attention, GELU MLP, mean pool, linear head (the baseline).
B   Born transformer: every block a ratio of polynomials of the token amplitudes a_t (a_t = 1/sqrt(n) for the n
    tokens present, 0 for absent positions). Embedding E w_t times a position phase, Born attention
    w_ij = |<q_i|k_j>|^2 / sum_j |<q_i|k_j>|^2, bilinear MLP Re[(W h) * conj(V h)], Born readout over N outcomes:
    P_b = |phi_b|^2 / sum_b |phi_b|^2, phi = sum_t R h_t (positions interfere; `pool="incoherent"` sums intensities
    instead). Complex by default; `complex_=False` is the same model in real numbers with the position phase made a
    rotation of dimension pairs (its real form).
P1  polynomial transformer, simulation only: unnormalised attention (Q K^T) V / n (odd degree, not measurable as a
    Born probability), bilinear MLP, no norm, mean pool, linear head.

Every denominator of B and P1 goes through `Frozen`: on a plain forward it is computed and recorded; on an explaining
forward the recorded values are reused, so the explained score is a polynomial of the input (as Z is held fixed in
Seeing Why). With no bias anywhere in B, that polynomial vanishes at a = 0.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# added to every denominator against a division by zero (no token present). With the denominators free and the
# amplitudes scaled towards zero (free_denominators.py), a row sum falls below 1e-30 near alpha ~ 1e-7 and the guard
# would take over; those scripts set GUARD = 1e-300.
GUARD = 1e-30


class Frozen:
    """Records denominators on the first pass, replays them on later passes."""

    def __init__(self):
        self.values, self.replay, self.i = [], False, 0

    def __call__(self, d):
        if not self.replay:
            self.values.append(d.detach())
            return d
        v = self.values[self.i]
        self.i += 1
        return v

    def start_replay(self):
        self.replay, self.i = True, 0
        return self


def _free(frozen):
    return frozen if frozen is not None else (lambda d: d)


# ----------------------------------------------------------------------------------------------------------------- S
class SoftmaxBlock(nn.Module):
    def __init__(self, d, heads, d_ff):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d))

    def forward(self, h, pad):
        z = self.ln1(h)
        h = h + self.att(z, z, z, key_padding_mask=pad, need_weights=False)[0]
        return h + self.mlp(self.ln2(h))


class SoftmaxTransformer(nn.Module):
    family = "S"

    def __init__(self, vocab, n_classes, max_len=32, d=64, heads=2, d_ff=256, blocks=1, readout="mean"):
        super().__init__()
        # readout (diagnostic): "mean" over the tokens, as every family; "last" the last token's state only
        self.readout = readout
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(SoftmaxBlock(d, heads, d_ff) for _ in range(blocks))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, n_classes)

    def embed(self, ids):
        x = self.emb(ids) + self.pos.weight[: ids.shape[1]]
        return x * (ids != 0).unsqueeze(-1)

    def from_embedded(self, x, ids):
        pad = ids == 0
        h = x
        for b in self.blocks:
            h = b(h, pad)
        if self.readout == "last":
            last = (~pad).sum(1).clamp(min=1) - 1
            return self.head(self.ln(h[torch.arange(len(h), device=h.device), last]))
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        return self.head(self.ln((h * keep).sum(1) / keep.sum(1).clamp(min=1)))

    def forward(self, ids):
        return self.from_embedded(self.embed(ids), ids)


# ---------------------------------------------------------------------------------------------------------------- P1
class PolyTransformer(nn.Module):
    family = "P1"

    def __init__(self, vocab, n_classes, max_len=32, d=64, heads=2, d_ff=170, blocks=1, attn="linear", ff="bilinear"):
        super().__init__()
        # attn (diagnostic): "linear" (q.k) v / n; "sq" (q.k)^2 v / n; "born" (q.k)^2 / sum_j (q.k)^2, Born-style;
        # "softmax" exp(q.k / sqrt(d_h)) / sum_j, the ablation. ff: "bilinear" (W1 h)(W2 h) W3, or "gelu"
        # W3 gelu(W1 h) (the ablation; d_ff 256 gives it the bilinear one's parameter count). Either ablation makes
        # the model non-polynomial: degree() is None and the evidence is integrated gradients with 256 nodes.
        assert attn in ("linear", "sq", "born", "softmax") and ff in ("bilinear", "gelu")
        self.d, self.heads, self.n_blocks, self.attn, self.ff = d, heads, blocks, attn, ff
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        s = 1 / math.sqrt(d)
        self.qkvo = nn.ParameterList(nn.Parameter(torch.randn(4, d, d) * s) for _ in range(blocks))
        self.wvu = nn.ParameterList(nn.Parameter(torch.randn(3, d_ff, d) / math.sqrt(d)) for _ in range(blocks))
        self.head = nn.Linear(d, n_classes, bias=False)

    def degree(self):
        if self.attn == "softmax" or self.ff == "gelu":
            return None
        deg = 1
        for _ in range(self.n_blocks):
            deg = {"linear": 3, "sq": 5, "born": 5}[self.attn] * deg     # (q.k) v, (q.k)^2 v
            deg = 2 * deg      # bilinear MLP
        return deg

    def embed(self, ids):
        x = self.emb(ids) + self.pos.weight[: ids.shape[1]]
        return x * (ids != 0).unsqueeze(-1)

    def from_embedded(self, x, ids, frozen=None):
        fz = _free(frozen)
        n = fz((ids != 0).sum(1).clamp(min=1).to(x.dtype))           # [B]
        h, dh = x, self.d // self.heads
        for qkvo, wvu in zip(self.qkvo, self.wvu):
            q, k, v = (torch.einsum("btd,ed->bte", h, m).unflatten(-1, (self.heads, dh)) for m in qkvo[:3])
            s = torch.einsum("bihe,bjhe->bhij", q, k)
            if self.attn == "softmax":
                s = (s / math.sqrt(dh)).masked_fill((ids == 0)[:, None, None], -1e9)    # padded keys: weight 0
                s = (s - s.amax(-1, keepdim=True).detach()).exp()
                s = s / (fz(s.sum(-1, keepdim=True)) + GUARD)
                # a padded slot stays empty, as in the polynomial attentions (there q = 0 gives y = 0 by itself)
                y = torch.einsum("bhij,bjhe->bihe", s, v).flatten(-2) * (ids != 0)[:, :, None].to(s.dtype)
            elif self.attn != "linear":
                s = s ** 2
            if self.attn == "born":
                s = s / (fz(s.sum(-1, keepdim=True)) + GUARD)
                y = torch.einsum("bhij,bjhe->bihe", s, v).flatten(-2)
            elif self.attn != "softmax":
                y = torch.einsum("bhij,bjhe->bihe", s, v).flatten(-2) / n[:, None, None]
            h = h + y @ qkvo[3].T
            if self.ff == "gelu":
                h = h + F.gelu(h @ wvu[0].T) @ wvu[2]
            else:
                h = h + ((h @ wvu[0].T) * (h @ wvu[1].T)) @ wvu[2]
        return self.head(h.sum(1) / n[:, None])

    def forward(self, ids):
        return self.from_embedded(self.embed(ids), ids)


# ----------------------------------------------------------------------------------------------------------------- B
class BornTransformer(nn.Module):
    family = "B"

    def __init__(self, vocab, n_classes, max_len=32, d=32, heads=2, d_ff=170, blocks=1, n_out=256,
                 complex_=True, pool="coherent", mean_channel=True, fixed_phase=False, spacing=0.1, attn="born", add_pos=False, amp_gain=1.0, lin_head=False, emb_init=None, no_phase=False, hw=False, gain0=10.0):
        super().__init__()
        assert d % heads == 0 and (complex_ or d % 2 == 0)
        self.d, self.heads, self.n_blocks, self.cx, self.pool = d, heads, blocks, complex_, pool
        self.mean_channel = mean_channel
        # amp_gain (diagnostic): token amplitudes times a constant; Born's ratios do not change with it, the
        # relative weight of the MLP's degree-2 term against the residual does
        self.amp_gain = amp_gain
        # hw: the shared hardware constraints (README, "Plan: hardware constraints"): every matrix a contraction,
        # every token state of norm <= 1, residuals as 50:50 combiners, and the only gain classical, one trained
        # scalar per block on the measured MLP numbers before they are re-encoded
        # (diagnostic: hw may be a string of the parts on - c contraction, n token norm, r combiner, g gain)
        self.hw = set("cnrg" if hw is True else hw or "")
        self.log_gain = nn.Parameter(torch.full((blocks,), math.log(gain0))) if "g" in self.hw else None
        # t (with r): a tunable beam splitter instead of 50:50, h <- cos(theta) h + sin(theta) update, one trained
        # angle per sublayer (attention, MLP) and block, starting at pi/4
        self.theta = nn.Parameter(torch.full((blocks, 2), math.pi / 4)) if "t" in self.hw else None
        # lin_head (diagnostic, not a Born readout): logits = Re[H sum_t h_t], trained on their cross-entropy
        self.lin_head = nn.Linear(d, n_classes, bias=False) if lin_head else None
        assert attn in ("born", "born2", "bhr")
        self.attn = attn
        self.mean_gain = nn.Parameter(torch.ones(())) if mean_channel else None
        self.max_len, self.n_out, self.n_classes = max_len, n_out, n_classes
        parts = 2 if complex_ else 1
        s = 1 / math.sqrt(parts * d)
        self.emb = nn.Parameter(torch.randn(parts, vocab, d) * (s if emb_init is None else emb_init))
        # optional: a learned field per position added to the word's (a position-controlled preparation), as the
        # additive positions of Softmax and Poly-sim; the phase stays
        self.pos = nn.Parameter(torch.randn(parts, max_len, d) * (s if emb_init is None else emb_init)) if add_pos else None
        self.no_phase = no_phase                                               # diagnostic: no position phase
        n_freq = d if complex_ else d // 2
        self.spacing = None
        if fixed_phase == "spacing":
            # a lens's linear grid of Fourier-plane frequencies, omega_k = s * k for k in [-1, 1]; one trained scalar s,
            # the word spacing relative to the lens's focal length and wavelength
            self.register_buffer("grid", torch.linspace(-1, 1, n_freq))
            self.spacing = nn.Parameter(torch.tensor(float(spacing)))
        elif fixed_phase:
            # what a lens gives a sentence laid out along one axis (shift theorem): the phase e^{-i k t} of a DFT
            # grid, k = 0..n-1 in steps of 2 pi / n, not trained
            self.register_buffer("omega", -2 * math.pi * torch.arange(n_freq, dtype=torch.float32) / n_freq)
        else:
            self.omega = nn.Parameter(torch.rand(n_freq) * 2 * math.pi / max_len)
        self.qkvo = nn.ParameterList(nn.Parameter(torch.randn(parts, 4, d, d) * s) for _ in range(blocks))
        self.wv = nn.ParameterList(nn.Parameter(torch.randn(parts, 2, d_ff, d) * s) for _ in range(blocks))
        self.u = nn.ParameterList(nn.Parameter(torch.randn(parts, d, d_ff) / math.sqrt(parts * d_ff))
                                  for _ in range(blocks))
        self.readout = nn.Parameter(torch.randn(parts, n_out, d) * s)
        groups = torch.tensor_split(torch.arange(n_out), n_classes)
        g = torch.zeros(n_classes, n_out)
        for c, idx in enumerate(groups):
            g[c, idx] = 1
        self.register_buffer("groups", g)                                  # [G, N] contiguous groups S_g
        self.register_buffer("share", g.sum(1) / n_out)                    # b_g = |S_g| / N

    def _c(self, p):
        return torch.complex(p[0], p[1]) if self.cx else p[0]

    def _mix(self, h, upd, blk, i, res):
        if self.theta is None:
            return res * (h + upd)
        th = self.theta[blk, i]
        return torch.cos(th) * h + torch.sin(th) * upd

    def _m(self, p):
        """A matrix (or a stack of them); under hw divided by its largest singular value when that exceeds 1, so it
        is a contraction: a block of a unitary on twice the modes."""
        m = self._c(p)
        if "c" not in self.hw:
            return m
        return m / torch.linalg.matrix_norm(m, ord=2).clamp(min=1)[..., None, None]

    def degree(self):
        deg = 1
        for _ in range(self.n_blocks):
            # |<q|k>|^2 v (or |<q|k>|^4 v) over a held denominator; with BHR attention the weights are held whole
            deg = {"born": 5, "born2": 9, "bhr": 1}[self.attn] * deg
            deg = 2 * deg      # bilinear MLP
        return 2 * deg         # Born readout

    def _bhr_weights(self, w, ids):
        """BHR on the measured key distribution: headroom over the no-information share 1/n of the n keys present
        (the reference slot counts), softmax(kappa LeakyReLU(u)) with kappa from pt 1.5 for n outcomes. Classical
        post-processing between blocks, no trained parameter; the explanation holds these weights whole."""
        present = ids != 0
        if self.mean_channel:
            present = torch.cat((present, torch.ones_like(present[:, :1])), 1)
        n = present.sum(1).to(w.dtype).clamp(min=2)[:, None, None, None]     # [B, 1, 1, 1]
        u = (w - 1 / n) / (1 - 1 / n)
        kappa = torch.log(0.99 * (n - 1) / 0.01) / (0.5 / (n - 1))
        z = (kappa * F.leaky_relu(u, 0.01)).masked_fill(~present[:, None, None, :], float("-inf"))
        return torch.softmax(z, -1) * present[:, None, :, None]

    def amplitudes(self, ids):
        present = (ids != 0).to(self.emb.dtype)
        return self.amp_gain * present / present.sum(1, keepdim=True).clamp(min=1).sqrt()

    def encode(self, ids, amp, fz):
        """Token states a_t * R(omega t) E w_t, [B, L, d]; zero where a_t = 0. With the mean channel, each token
        keeps its contrast a_t e_t - A_t m and one more slot (position value L) holds the reference
        g * m, m = sum_t a_t e_t / sum_t a_t: [B, L + 1, d]. The denominator and the coefficient A_t (= a_t on a
        plain forward) are held, so the encoding stays linear in a; a token with a_t = 0 vanishes from both."""
        e = self._c(self.emb)[ids]
        if self.pos is not None:
            e = e + self._c(self.pos)[: ids.shape[1]]
        if "n" in self.hw:
            e = e / torch.linalg.vector_norm(e, dim=-1, keepdim=True).clamp(min=1)    # a field of power <= 1
        omega = self.spacing * self.grid if self.spacing is not None else self.omega
        if self.no_phase:
            omega = torch.zeros_like(omega)
        t = torch.arange(ids.shape[1], device=ids.device, dtype=omega.dtype)
        ang = t[:, None] * omega[None, :]
        if self.cx:
            e = e * torch.polar(torch.ones_like(ang), ang)
        else:
            x1, x2 = e[..., 0::2], e[..., 1::2]
            c, s = torch.cos(ang), torch.sin(ang)
            e = torch.stack((c * x1 - s * x2, s * x1 + c * x2), -1).flatten(-2)
        a = amp.unsqueeze(-1).to(e.dtype)
        if not self.mean_channel:
            return e * a
        m = (e * a).sum(1) / (fz(amp.sum(1)) + GUARD)[:, None].to(e.dtype)    # [B, d]
        coef = fz(amp).unsqueeze(-1).to(e.dtype)
        return torch.cat((e * a - coef * m[:, None], (self.mean_gain * m)[:, None]), 1)

    def numerators(self, ids, amp, frozen=None):
        """(N_g, Z): N_g = sum_b (Pi_g - b_g)_b |phi_b|^2 and Z = sum_b |phi_b|^2 with every inner denominator
        passed through `frozen`; P_g - b_g = N_g / Z."""
        fz = _free(frozen)
        h, dh = self.encode(ids, amp, fz), self.d // self.heads
        res = 1 / math.sqrt(2) if "r" in self.hw else 1.0                             # a 50:50 combiner under hw
        for blk, (qkvo, wv, u) in enumerate(zip(self.qkvo, self.wv, self.u)):
            m = self._m(qkvo)
            q, k, v = (torch.einsum("btd,ed->bte", h, m[i]).unflatten(-1, (self.heads, dh)) for i in range(3))
            ov = torch.einsum("bihe,bjhe->bhij", q.conj(), k)
            s = ov.real ** 2 + ov.imag ** 2 if self.cx else ov ** 2           # |<q_i|k_j>|^2
            if self.attn == "born2":
                s = s ** 2                                                    # |<q_i|k_j>|^4: two copies
            den = fz(s.sum(-1, keepdim=True))
            w = s / (den + GUARD)
            if self.attn == "bhr":
                w = fz(self._bhr_weights(w, ids))
            y = torch.einsum("bhij,bjhe->bihe", w.to(v.dtype), v).flatten(-2)
            h = self._mix(h, y @ m[3].T, blk, 0, res)
            w = self._m(wv)
            a, b = h @ w[0].T, h @ w[1].T
            mid = (a * b.conj()).real if self.cx else a * b                  # Re[(W h) * conj(V h)]
            if self.log_gain is not None:
                mid = mid * self.log_gain[blk].exp()                           # classical gain, after measuring
            h = self._mix(h, mid.to(h.dtype) @ self._m(u).T, blk, 1, res)
        if self.lin_head is not None:
            pooled = h.sum(1)
            return self.lin_head((pooled.real if self.cx else pooled).to(self.lin_head.weight.dtype)), None
        r = h @ self._m(self.readout).T                                        # [B, L, N]
        if self.pool == "coherent":
            phi = r.sum(1)
            inten = phi.real ** 2 + phi.imag ** 2 if self.cx else phi ** 2
        else:
            inten = (r.real ** 2 + r.imag ** 2 if self.cx else r ** 2).sum(1)
        z = inten.sum(-1)
        mass = inten @ self.groups.T                                          # [B, G]
        num = mass - self.share * z[:, None]
        return num, fz(z)

    def class_probs(self, ids):
        num, z = self.numerators(ids, self.amplitudes(ids))
        if self.lin_head is not None:
            return num                                                         # logits
        return num / (z[:, None] + GUARD) + self.share                         # P_g

    def forward(self, ids):
        return self.class_probs(ids)


class CoherentBorn(nn.Module):
    """C: no measurement until the end. Every token's field a_t (E w_t + P_t + S_seg) e^{i omega t} (plus a reference
    slot of amplitude 1) is prepared in two copies; Q acts on the first, conj(K) on the second; the slots are summed
    coherently (a combiner) and a linear map R on the two-copy space, phi = R (q_bar * conj(k_bar)), is measured once
    by the camera: P_g = |Pi_g phi|^2 / |phi|^2. The only normalisation is Z, the only nonlinearity the final
    measurement, so the model is a degree-4 polynomial of the amplitudes over one held denominator and its evidence
    is exact as B's. A sum over all pairs factorises into the two pooled copies, so the segment (premise or
    hypothesis) is added to each field: without it the pooled state cannot tell the two sentences apart.
    Simulation of the circuit; it is not compiled by compile_hw (compilable = False)."""
    family = "B"
    compilable = False
    mean_channel = False
    lin_head = None

    def __init__(self, vocab, n_classes, max_len=32, d=32, r=128, n_out=256, sep=2):
        super().__init__()
        self.d, self.r, self.n_out, self.sep = d, r, n_out, sep
        s = 1 / math.sqrt(2 * d)
        self.emb = nn.Parameter(torch.randn(2, vocab, d))                    # embeddings at scale 1 (round 4)
        self.pos = nn.Parameter(torch.randn(2, max_len, d))
        self.seg = nn.Parameter(torch.randn(2, 2, d))
        self.ref = nn.Parameter(torch.randn(2, d) * s)
        self.omega = nn.Parameter(torch.rand(d) * 2 * math.pi / max_len)
        self.q = nn.Parameter(torch.randn(2, r, d) * s)
        self.k = nn.Parameter(torch.randn(2, r, d) * s)
        self.readout = nn.Parameter(torch.randn(2, n_out, r) / math.sqrt(2 * r))
        groups = torch.tensor_split(torch.arange(n_out), n_classes)
        g = torch.zeros(n_classes, n_out)
        for c, idx in enumerate(groups):
            g[c, idx] = 1
        self.register_buffer("groups", g)
        self.register_buffer("share", g.sum(1) / n_out)

    @staticmethod
    def _c(p):
        return torch.complex(p[0], p[1])

    def degree(self):
        return 4

    def amplitudes(self, ids):
        present = (ids != 0).to(self.emb.dtype)
        return present / present.sum(1, keepdim=True).clamp(min=1).sqrt()

    def numerators(self, ids, amp, frozen=None):
        fz = _free(frozen)
        L = ids.shape[1]
        seg = (torch.cumsum((ids == self.sep).long(), 1) > 0).long()
        e = self._c(self.emb)[ids] + self._c(self.pos)[:L] + self._c(self.seg)[seg]
        t = torch.arange(L, device=ids.device, dtype=self.omega.dtype)
        e = e * torch.polar(torch.ones_like(t[:, None] * self.omega), t[:, None] * self.omega)
        x = e * amp.unsqueeze(-1).to(e.dtype)
        ref = self._c(self.ref)
        q_bar = x.sum(1) @ self._c(self.q).T + ref @ self._c(self.q).T          # [B, r]: slots summed coherently
        k_bar = x.sum(1) @ self._c(self.k).T + ref @ self._c(self.k).T
        phi = (q_bar * k_bar.conj()) @ self._c(self.readout).T                 # [B, n_out]
        inten = phi.real ** 2 + phi.imag ** 2
        z = inten.sum(-1)
        mass = inten @ self.groups.T
        return mass - self.share * z[:, None], fz(z)

    def forward(self, ids):
        num, z = self.numerators(ids, self.amplitudes(ids))
        return num / (z[:, None] + GUARD) + self.share


class CoherentAttn(CoherentBorn):
    """C with attention, still no measurement until the end: three copies of the sentence state. The first is summed
    over its slots into a pooled query q_bar; the second carries conj(k_j), the third v_j, and a projector pairs their
    slot registers (j = j'). Contracting the first two copies' features gives the amplitude s_j = <k_j|q_bar> per slot,
    so y = sum_j s_j v_j: content-based attention, unnormalised (a division needs a measurement). Several heads; the
    camera reads phi = R y once. Degree 6 over the one held denominator Z: evidence exact."""

    def __init__(self, vocab, n_classes, max_len=32, d=32, heads=4, dk=16, dv=32, n_out=256, sep=2):
        super().__init__(vocab, n_classes, max_len, d=d, r=1, n_out=n_out, sep=sep)
        del self.q, self.k, self.readout
        s = 1 / math.sqrt(2 * d)
        self.heads, self.dk, self.dv = heads, dk, dv
        self.wq = nn.Parameter(torch.randn(2, heads, dk, d) * s)
        self.wk = nn.Parameter(torch.randn(2, heads, dk, d) * s)
        self.wv = nn.Parameter(torch.randn(2, heads, dv, d) * s)
        self.readout = nn.Parameter(torch.randn(2, n_out, heads * dv) / math.sqrt(2 * heads * dv))

    def degree(self):
        return 6

    def values(self, ids, amp):
        """y = sum_j <k_j|q_bar> v_j per head, flattened [B, heads * dv] (index v + dv h)."""
        L = ids.shape[1]
        seg = (torch.cumsum((ids == self.sep).long(), 1) > 0).long()
        e = self._c(self.emb)[ids] + self._c(self.pos)[:L] + self._c(self.seg)[seg]
        t = torch.arange(L, device=ids.device, dtype=self.omega.dtype)
        e = e * torch.polar(torch.ones_like(t[:, None] * self.omega), t[:, None] * self.omega)
        x = e * amp.unsqueeze(-1).to(e.dtype)
        x = torch.cat((x, self._c(self.ref).expand(x.shape[0], 1, -1)), 1)    # the reference slot, amplitude 1
        q_bar = torch.einsum("btd,hkd->bhk", x, self._c(self.wq))              # first copy, slots summed
        k = torch.einsum("btd,hkd->bhtk", x, self._c(self.wk))                 # second copy, per slot
        v = torch.einsum("btd,hvd->bhtv", x, self._c(self.wv))                 # third copy, per slot
        s = torch.einsum("bhtk,bhk->bht", k.conj(), q_bar)                     # <k_j|q_bar>
        return torch.einsum("bht,bhtv->bhv", s, v).flatten(1)

    def numerators(self, ids, amp, frozen=None):
        fz = _free(frozen)
        phi = self.values(ids, amp) @ self._c(self.readout).T
        inten = phi.real ** 2 + phi.imag ** 2
        z = inten.sum(-1)
        mass = inten @ self.groups.T
        return mass - self.share * z[:, None], fz(z)


def brick_pairs(nq, layers):
    """Neighbour pairs (a, a + 1) of each layer of a brickwall on nq qubits in a line."""
    return [[(a, a + 1) for a in range(l % 2, nq - 1, 2)] for l in range(layers)]


def brick_gates(gen):
    """[layers, g, 4, 4] unitaries exp(A - A^H) from generators [layers, g, 2, 4, 4] (re, im); a gate on (a, a + 1)
    has index b_a + 2 b_(a+1)."""
    a = torch.complex(gen[:, :, 0], gen[:, :, 1])
    return torch.linalg.matrix_exp(a - a.conj().transpose(-1, -2))


def brick_apply(gates, pairs, nq, y):
    """The brickwall applied to y [B, 2^nq] (qubit q is bit q of the index), gate by gate."""
    B = y.shape[0]
    psi = y.reshape([B] + [2] * nq)                                           # axis 1 + k <-> qubit nq - 1 - k
    for l, layer in enumerate(pairs):
        for i, (a, b) in enumerate(layer):
            ax = (nq - b, nq - a)                                             # (b_(a+1), b_a) last: index 2 b_(a+1) + b_a
            t = psi.movedim(ax, (-2, -1))
            sh = t.shape
            t = (t.reshape(-1, 4) @ gates[l, i].T).reshape(sh)
            psi = t.movedim((-2, -1), ax)
    return psi.reshape(B, -1)


def brick_matrix(gates, pairs, nq, like):
    """The brickwall as a 2^nq x 2^nq matrix (dtype and device of `like`), in a few batched kernels: each layer is
    a Kronecker product of its gates (an even layer leaves the top qubit alone, an odd one the bottom), and the
    layers are multiplied pairwise in a tree. brick_apply on the identity gives the same matrix, gate by gate."""
    assert nq == 7 and pairs == brick_pairs(nq, len(pairs))
    g = gates.to(like.dtype)
    eye2 = torch.eye(2, dtype=g.dtype, device=g.device).expand(len(g), 2, 2)
    k3 = lambda a, b, c: torch.einsum("lij,lkm,lno->liknjmo", a, b, c).reshape(-1, 64, 64)
    k2 = lambda a, b: torch.einsum("lij,lkm->likjm", a, b).reshape(-1, a.shape[1] * b.shape[1],
                                                                       a.shape[2] * b.shape[2])
    even = k2(eye2, k3(g[:, 2], g[:, 1], g[:, 0]))                          # q6 | (q5 q4) (q3 q2) (q1 q0)
    odd = k2(k3(g[:, 2], g[:, 1], g[:, 0]), eye2)                           # (q6 q5) (q4 q3) (q2 q1) | q0
    layer = torch.arange(len(g), device=g.device)
    m = torch.where((layer % 2 == 0)[:, None, None], even, odd)             # [layers, 128, 128], layer 0 first
    while len(m) > 1:
        rest = m[-1:] if len(m) % 2 else m[:0]
        n = len(m) - len(rest)
        m = torch.cat((m[1:n:2] @ m[0:n:2], rest))                             # later @ earlier
    return m[0]


class CoherentAttnU(CoherentAttn):
    """Coherent-attn with a cheap readout: R is a trained brickwall of two-qubit unitaries on the 7 qubits of y (5 value
    qubits, then 2 head qubits; y index = v + 32 h), then all 7 qubits are read and the 128 outcomes are split into the
    class groups. A unitary needs no ancilla and loses no scale, and each two-qubit gate costs at most 3 CNOTs between
    neighbours on a line, so the readout is 3 * 3 * layers CNOTs instead of a dense 9-qubit block encoding. Same
    polynomial degree and held denominator Z = |y|^2: evidence exact."""

    def __init__(self, vocab, n_classes, max_len=32, layers=12, **kw):
        super().__init__(vocab, n_classes, max_len, n_out=128, **kw)
        assert self.heads * self.dv == 128
        del self.readout
        self.nq, self.layers = 7, layers
        self.pairs = brick_pairs(self.nq, layers)
        self.gen = nn.Parameter(torch.randn(layers, 3, 2, 4, 4) * 0.5)        # generators of each layer's 3 gates

    def gates(self):
        return brick_gates(self.gen)

    def apply_readout(self, y):
        """U y for y [B, 128] (qubit q is bit q of the index): the gates multiplied into one 128 x 128 matrix first
        (on the identity), then one product, so a step runs a few hundred tiny kernels once, not per batch row."""
        return y @ brick_matrix(self.gates(), self.pairs, self.nq, y).T

    def readout_matrix(self):
        g = self.gates()
        return brick_matrix(g, self.pairs, self.nq, g)

    def numerators(self, ids, amp, frozen=None):
        fz = _free(frozen)
        y = self.values(ids, amp)
        phi = self.apply_readout(y)
        inten = phi.real ** 2 + phi.imag ** 2
        z = inten.sum(-1)
        mass = inten @ self.groups.T
        return mass - self.share * z[:, None], fz(z)


class CoherentAttnB(CoherentAttnU):
    """Coherent-attn with every map a brickwall of two-qubit unitaries, the head register (2 qubits, |++>) shared by
    the two of them. U_M on (copy A's features, head) turns x_bar into m = U_M (x_bar (x) |++>); the Bell projection
    of A with copy B (conj x_j) gives c_(j,h) = <x_j|m_h>, a query-key overlap per slot and head; copy C (x_j) then
    carries y = sum_j x_j (x) c_j on (C's features, head), and one brickwall U (values and readout in one) maps it
    to the 7 qubits read. No block encoding, so no ancilla and no scale lost; same degree 6 over Z = |U y|^2 = |y|^2,
    evidence exact. In the circuit: 17 + p qubits (B and C paired), CNOTs 9 (layers_m + layers) + projections + the
    preparations."""

    def __init__(self, vocab, n_classes, max_len=32, layers_m=12, layers=24):
        super().__init__(vocab, n_classes, max_len, layers=layers)
        del self.wq, self.wk, self.wv
        self.layers_m = layers_m
        self.pairs_m = brick_pairs(self.nq, layers_m)
        self.mgen = nn.Parameter(torch.randn(layers_m, 3, 2, 4, 4) * 0.5)

    def m_gates(self):
        return brick_gates(self.mgen)

    def values(self, ids, amp):
        """y = sum_j x_j (x) c_j, index v + 32 h."""
        L = ids.shape[1]
        seg = (torch.cumsum((ids == self.sep).long(), 1) > 0).long()
        e = self._c(self.emb)[ids] + self._c(self.pos)[:L] + self._c(self.seg)[seg]
        t = torch.arange(L, device=ids.device, dtype=self.omega.dtype)
        e = e * torch.polar(torch.ones_like(t[:, None] * self.omega), t[:, None] * self.omega)
        x = e * amp.unsqueeze(-1).to(e.dtype)
        x = torch.cat((x, self._c(self.ref).expand(x.shape[0], 1, -1)), 1)    # the reference slot, amplitude 1
        B, D = x.shape[0], self.d
        a = (x.sum(1)[:, None, :] / 2).expand(B, 4, D).reshape(B, -1)       # x_bar (x) |++>, index a + 32 h'
        m = (a @ brick_matrix(self.m_gates(), self.pairs_m, self.nq, a).T).view(B, 4, D)
        c = torch.einsum("bta,bha->bth", x.conj(), m)                         # Bell projection of A with B
        return torch.einsum("btv,bth->bhv", x, c).flatten(1)


class BHR(nn.Module):
    """Born headroom readout (Seeing Why, Eq. 2): softmax(kappa LeakyReLU((P_g - b_g)/(1 - b_g))); returns log p."""
    EPS = ALPHA = 0.01

    def __init__(self, share, pt=1.5):
        super().__init__()
        g = len(share)
        self.register_buffer("share", share)
        self.kappa = math.log((1 - self.EPS) * (g - 1) / self.EPS) / ((pt - 1) / (g - 1))

    def forward(self, probs):
        u = (probs - self.share) / (1 - self.share)
        return F.log_softmax(self.kappa * F.leaky_relu(u, self.ALPHA), -1)


def build(name, vocab, n_classes, blocks=1, max_len=32):
    if name == "Coherent-attn":
        return CoherentAttn(vocab, n_classes, max_len)
    if name.startswith("Coherent-attn-b"):                                   # Coherent-attn-b<layers_m>-<layers>
        lm, l = name[len("Coherent-attn-b"):].split("-")
        return CoherentAttnB(vocab, n_classes, max_len, layers_m=int(lm), layers=int(l))
    if name.startswith("Coherent-attn-u"):                                   # Coherent-attn-u<layers>
        return CoherentAttnU(vocab, n_classes, max_len, layers=int(name[len("Coherent-attn-u"):] or 12))
    if name == "Coherent" or name.startswith("Coherent-r"):
        return CoherentBorn(vocab, n_classes, max_len, r=int(name.split("-r")[1]) if "-r" in name else 128)
    if name == "Softmax":
        return SoftmaxTransformer(vocab, n_classes, max_len, blocks=blocks)
    if name == "Softmax-last":
        return SoftmaxTransformer(vocab, n_classes, max_len, blocks=blocks, readout="last")
    if name == "Poly-sim":
        return PolyTransformer(vocab, n_classes, max_len, blocks=blocks)
    if name in ("Poly-softmax-bil", "Poly-born-gelu", "Poly-softmax-gelu"):
        # ablations of Poly-sim-born: which part gives the evidence its quality, the attention or the feed-forward
        _, a, f = name.split("-")
        return PolyTransformer(vocab, n_classes, max_len, blocks=blocks, attn=a,
                               ff="gelu" if f == "gelu" else "bilinear", d_ff=256 if f == "gelu" else 170)
    if name in ("Poly-sim-sq", "Poly-sim-born"):
        return PolyTransformer(vocab, n_classes, max_len, blocks=blocks, attn=name.split("-")[-1])
    if name == "Born":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True)
    if name == "Born-camera":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, pool="incoherent")
    if name == "Born-dft":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, fixed_phase=True)
    if name == "Born-lens-0.1":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, fixed_phase="spacing")
    if name == "Born-lens-0.5":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, fixed_phase="spacing",
                               spacing=0.5)
    if name == "Born-p2":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, attn="born2")
    if name == "Born-bhr-attn":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, attn="bhr")
    if name == "Born-addpos":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, add_pos=True)
    if name == "Born-p2-addpos":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, attn="born2", add_pos=True)
    if name.startswith("Born-hw-"):
        # diagnostic: Born-hw-<parts on>[@<starting gain>], parts from c n r g t (see BornTransformer)
        parts, _, g0 = name[8:].partition("@")
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, add_pos=True,
                               emb_init=1.0, hw=parts or "-", gain0=float(g0 or 10))
    if name in ("Born-hw", "Born-p2-hw"):
        # Born-ap / Born-p2-ap under the shared hardware constraints
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, add_pos=True,
                               emb_init=1.0, attn="born2" if "p2" in name else "born", hw=True)
    if name.startswith(("Born-ap-d", "Born-p2-ap-d")):
        # the same with a smaller state, for the qubit profile: Born-ap-d<d>, Born-p2-ap-d<d>
        return BornTransformer(vocab, n_classes, max_len, d=int(name.rsplit("-d", 1)[1]), blocks=blocks,
                               complex_=True, add_pos=True, emb_init=1.0, attn="born2" if "p2" in name else "born")
    if name in ("Born-ap", "Born-p2-ap"):
        # the RECALL recipe (round 4): additive position fields (a fixed field per time bin) and embeddings started
        # at scale 1; mean channel, phases and coherent readout kept
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, add_pos=True,
                               emb_init=1.0, attn="born2" if "p2" in name else "born")
    if name == "Born-gain4":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, amp_gain=4.0)
    if name == "Born-real-64-linhead":
        return BornTransformer(vocab, n_classes, max_len, d=64, blocks=blocks, complex_=False, lin_head=True)
    if name == "Born-linhead":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, lin_head=True)
    if name.startswith("Diag-"):
        # Born-real-64 moved towards Poly-sim-born; each flag names one difference undone: Diag-<flags>, flags from
        # m (mean channel off), p (additive positions), z (no phase), g (amplitude gain 4), e (embedding init 1),
        # l (linear head), c (incoherent pooling), x (complex, d 32), 2 (|<q|k>|^4 attention)
        f = name[5:]
        return BornTransformer(vocab, n_classes, max_len, d=32 if "x" in f else 64, blocks=blocks, complex_="x" in f,
                               pool="incoherent" if "c" in f else "coherent", mean_channel="m" not in f,
                               add_pos="p" in f, no_phase="z" in f, amp_gain=4.0 if "g" in f else 1.0,
                               emb_init=1.0 if "e" in f else None, lin_head="l" in f,
                               attn="born2" if "2" in f else "born")
    if name == "Born-no-ref":
        return BornTransformer(vocab, n_classes, max_len, d=32, blocks=blocks, complex_=True, mean_channel=False)
    if name == "Born-real-64":
        return BornTransformer(vocab, n_classes, max_len, d=64, blocks=blocks, complex_=False)
    if name == "Born-real-46":
        return BornTransformer(vocab, n_classes, max_len, d=46, blocks=blocks, complex_=False)
    raise ValueError(name)


def count(model):
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for n, p in model.named_parameters() if n.startswith(("emb", "pos")))
    return total, total - emb
