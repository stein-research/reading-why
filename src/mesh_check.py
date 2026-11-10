"""Check the optical compile with an outside implementation: every fixed matrix as Clements meshes of Mach-Zehnder interferometers.

python mesh_check.py <tag> [--n 1000] [--sigma 0 0.01 0.02 0.05 0.1] [--draws 2]

Each fixed matrix M of the compiled pass (Q, K, V, O, the MLP's W, V and U of every block, the readout) is built as
an interferometer the way a chip would: M = s U diag(S) V^H (SVD), U and V^H each a full square mesh (full SVD: a 170 x 32 matrix gets a
170-mode mesh on one side) in the Clements layout of Mach-Zehnder interferometers (theta, phi) plus output phases, found by William R. Clements' own `interferometer` package (MIT;
`square_decomposition`), S attenuators and s the classical gain compile_hw already puts on measured numbers. The
meshes are rebuilt from their phases, optionally with Gaussian phase errors (sigma, in radians, on every theta, phi
and output phase), and the model with the rebuilt matrices runs through compile_hw.compiled_probs (noise-free
otherwise) on the first n test pairs. Gains follow the built matrices (a device is calibrated on what it built).

The rebuild multiplies the 2 x 2 interferometers onto two rows at a time (the package's `calculate_transformation`
forms an N x N matrix per interferometer, minutes at N = 170); it is checked against `calculate_transformation` on a
32-mode mesh with errors. Decompositions are cached in results/<tag>_mesh.npz (the 256-mode readout takes ~25 min).
"""
import json
import os
import sys
import time

import numpy as np
import torch
from interferometer.main import Interferometer, Beamsplitter, square_decomposition

from compile_hw import compiled_probs, load

args = sys.argv[1:]
tag = args[0]
opt = lambda k, dflt: args[args.index(k) + 1] if k in args else dflt
n_test, draws = int(opt("--n", 1000)), int(opt("--draws", 2))


def values(flag):
    """The numbers after `flag`, up to the next option (a later option's value is not one of them)."""
    out = []
    for a in args[args.index(flag) + 1:]:
        if a.startswith("--"):
            break
        out.append(float(a))
    return out


sigmas = values("--sigma") if "--sigma" in args else [0, 0.01, 0.02, 0.05, 0.1]

model, d = load(tag)                                                      # float64, CPU
C = lambda p: (p[0] + 1j * p[1]).detach().numpy()
# (parameter, index into its leading axes) for every fixed matrix of the compiled pass
slots = [("readout", ())]
for b in range(len(model.qkvo)):
    slots += [(f"qkvo.{b}", (i,)) for i in range(4)] + [(f"wv.{b}", (i,)) for i in range(2)] + [(f"u.{b}", ())]
params = dict(model.named_parameters())


def mesh_arrays(I, N):
    bs = np.array([(b.mode1, b.mode2, b.theta, b.phi) for b in I.BS_list], dtype=float).reshape(-1, 4)
    ph = np.zeros(N)
    ph[:len(I.output_phases)] = I.output_phases
    return bs, ph


def rebuild(bs, ph, N, sigma=0.0, rng=None):
    """The mesh's unitary from its phases (the package's convention), with Gaussian phase errors."""
    th, phi, out = bs[:, 2].copy(), bs[:, 3].copy(), ph.copy()
    if sigma:
        th += rng.normal(0, sigma, th.shape)
        phi += rng.normal(0, sigma, phi.shape)
        out += rng.normal(0, sigma, out.shape)
    U = np.eye(N, dtype=complex)
    for (m1, m2), t, p in zip(bs[:, :2].astype(int) - 1, th, phi):
        r1, r2 = U[m1].copy(), U[m2].copy()
        U[m1] = np.exp(1j * p) * np.cos(t) * r1 - np.sin(t) * r2
        U[m2] = np.exp(1j * p) * np.sin(t) * r1 + np.cos(t) * r2
    return np.exp(1j * out)[:, None] * U


# the fast rebuild against the package's own, on a 32-mode mesh with errors
rng = np.random.default_rng(0)
Ur = np.linalg.qr(rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32)))[0]
bs, ph = mesh_arrays(square_decomposition(Ur), 32)
bs_err = bs.copy()
bs_err[:, 2:] += rng.normal(0, 0.05, bs_err[:, 2:].shape)
I = Interferometer()
for m1, m2, t, p in bs_err:
    I.add_BS(Beamsplitter(int(m1), int(m2), t, p))
I.output_phases = list(ph)
ref = I.calculate_transformation()
mine = rebuild(np.column_stack((bs_err[:, :2], bs_err[:, 2:])), ph, 32)
check = dict(rebuild_vs_package=float(np.abs(mine - ref).max()), decompose_32=float(np.abs(rebuild(bs, ph, 32) - Ur).max()))
print(f"rebuild = package's calculate_transformation to {check['rebuild_vs_package']:.1e}; "
      f"32-mode decomposition rebuilt to {check['decompose_32']:.1e}", flush=True)

cache = f"results/{tag}_mesh.npz"
store = dict(np.load(cache)) if os.path.exists(cache) else {}
meshes = []
for name, ix in slots:
    M = C(params[name])[ix] if ix else C(params[name])
    s = np.linalg.norm(M, 2)
    U, S, Vh = np.linalg.svd(M / s)
    k = len(S)
    key = f"{name}_{'_'.join(map(str, ix))}"
    for side, W in (("U", U), ("V", Vh)):
        if f"{key}_{side}_bs" not in store:
            t0 = time.time()
            store[f"{key}_{side}_bs"], store[f"{key}_{side}_ph"] = mesh_arrays(square_decomposition(W), len(W))
            np.savez(cache, **store)
            print(f"  {key} {side}: {len(W)} modes, {time.time() - t0:.0f} s", flush=True)
    meshes.append((name, ix, M, s, S, k, key, len(U), len(Vh)))

ids = torch.tensor(d["x_test"][:n_test])
y = torch.tensor(d["y_test"][:n_test])


def run():
    with torch.no_grad():
        return torch.cat([compiled_probs(model, ids[i:i + 256])[0] for i in range(0, len(ids), 256)])


p0 = run()
rows = []
for sigma in sigmas:
    for draw in range(draws if sigma else 1):
        rng = np.random.default_rng(1000 + draw)
        err = 0.0
        for name, ix, M, s, S, k, key, nu, nv in meshes:
            Ub = rebuild(store[f"{key}_U_bs"], store[f"{key}_U_ph"], nu, sigma, rng)
            Vb = rebuild(store[f"{key}_V_bs"], store[f"{key}_V_ph"], nv, sigma, rng)
            Mb = s * (Ub[:, :k] * S) @ Vb[:k]
            err = max(err, float(np.abs(Mb - M).max() / np.abs(M).max()))
            p = params[name]
            with torch.no_grad():
                (p[0][ix] if ix else p[0]).copy_(torch.tensor(Mb.real))
                (p[1][ix] if ix else p[1]).copy_(torch.tensor(Mb.imag))
        pb = run()
        rows.append(dict(sigma=sigma, draw=draw, matrix_err_max=err, acc=float((pb.argmax(1) == y).float().mean()),
                         agree=float((pb.argmax(1) == p0.argmax(1)).float().mean()),
                         p_diff_max=float((pb - p0).abs().max()), p_diff_median=float((pb - p0).abs().median())))
        r = rows[-1]
        print(f"sigma {sigma:.3f} draw {draw}: matrices off by {err:.1e} (relative), accuracy {r['acc']:.4f} "
              f"(compiled {float((p0.argmax(1) == y).float().mean()):.4f}), answers kept {r['agree']:.4f}, "
              f"|dP| median {r['p_diff_median']:.1e} max {r['p_diff_max']:.1e}", flush=True)
for name, ix, M, *_ in meshes:                                            # put the trained matrices back
    p = params[name]
    with torch.no_grad():
        (p[0][ix] if ix else p[0]).copy_(torch.tensor(M.real))
        (p[1][ix] if ix else p[1]).copy_(torch.tensor(M.imag))
with open(f"results/{tag}_mesh.json", "w") as f:
    json.dump(dict(tag=tag, n=n_test, check=check, rows=rows,
                   beam_splitters=int(sum(len(store[k]) for k in store if k.endswith("_bs")))), f, indent=1)
