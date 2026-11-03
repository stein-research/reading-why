"""Exact per-token evidence: zero-baseline Integrated Gradients by Gauss-Legendre quadrature.

With every denominator held at the explained input (models.Frozen), the score of B and P1 is a polynomial f of the
input of degree D, so grad f(t x) is a polynomial of degree D - 1 in t and ceil(D / 2) Gauss-Legendre nodes give
IG = x * int_0^1 grad f(t x) dt exactly (up to float64). The token evidence then sums to f(x) - f(0), and f(0) = 0
for B (P_g - b_g = N_g / Z) and P1 (no bias). For B the inputs are the token amplitudes a_t, one number per token;
for S and P1 they are the embedded token vectors, summed over their dimensions. S is not a polynomial, and its
LayerNorm makes it scale-invariant, so f jumps at the zero baseline: its IG sums to f(x) - lim_{t->0} f(t x), not to
f(x) - f(0), and the gap is reported.
"""
import math

import numpy as np
import torch

from models import Frozen


def nodes01(n):
    t, w = np.polynomial.legendre.leggauss(n)
    return (t + 1) / 2, w / 2


def _score_fn(model, ids):
    """(x0, f) with f(x, frozen) -> scores [B, C]."""
    if model.family == "B":
        def f(x, fz):
            num, z = model.numerators(ids, x, fz)
            return num / (z[:, None] + 1e-300)
        return model.amplitudes(ids), f
    if model.family == "P1":
        return model.embed(ids), lambda x, fz: model.from_embedded(x, ids, fz)
    return model.embed(ids), lambda x, fz: model.from_embedded(x, ids)


def evidence(model, ids, target, n_nodes=None):
    """Per-token evidence [B, L] for class `target` [B], the explained score f(x) [B] and f(0) [B]."""
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    if n_nodes is None:
        deg = model.degree() if hasattr(model, "degree") else None
        n_nodes = math.ceil(deg / 2) if deg else 256
    fz = Frozen()
    with torch.no_grad():
        full = f(x0, fz).gather(1, target[:, None])[:, 0]
        zero = f(torch.zeros_like(x0), fz.start_replay()).gather(1, target[:, None])[:, 0]
    total = torch.zeros_like(x0)
    for t, w in zip(*nodes01(n_nodes)):
        x = (float(t) * x0).requires_grad_(True)
        out = f(x, fz.start_replay()).gather(1, target[:, None]).sum()
        (g,) = torch.autograd.grad(out, x)
        total += float(w) * g
    ig = x0 * total
    ev = ig if ig.dim() == 2 else ig.sum(-1)
    return ev, full, zero
