"""Attention read by quantum circuits under an IBM device's noise model (simulated; no device used).

python hw_attention.py <tag> [--index 15] [--backend FakeTorino] [--shots 10000]
python hw_attention.py <tag> --n 50 [--backend FakeTorino] [--shots 10000]   # first n test pairs, noisy only
python hw_attention.py <tag> --evidence 20 [...]   # first n test pairs: evidence with the measured denominators
python hw_attention.py <tag> --index 3763 --shots 2000 --device ibm_kingston --account 1 [--submit]   # an IBM device,
#   through the ledger (ibm_jobs/); without --submit a dry run: transpiled and recorded, not sent;
#   --offline: no account and no ledger - every circuit transpiled for and run on the --backend noise model
#   --noisy-dry: the dry run goes on with counts from the --backend noise model instead of ideal ones;
#   --pilot 30: only the first 30 circuits of block 0 (repeat 1 in the ledger), time and visibility, then stop;
#   --eta v: use visibility v for every row instead of the self-overlap circuits (which the transpiler makes shallow:
#   |+>|q> is a product state, 13 CZ against 37), e.g. the pilot's slope;
#   --delete p --no-evidence: the deletion test on the device - word p removed, the answer only (no calibrations)
#   --cal-random n: instead of the self-overlap rows, n calibration circuits per block as deep as the overlaps
#   (Hadamard tests of random state pairs with known overlap); one slope and one readout offset per block are fitted
#   to them, the offset is taken off every overlap and the row sums are divided by the slope's fourth power

Evidence on hardware (hybrid): the evidence is the exact polynomial's with its denominators held (models.Frozen);
here the attention row sums sum_j |<q_i|k_j>|^4 are the ones the circuits measured, the numerators stay exact (on
the computer). The device's loss of visibility eta shrinks every measured overlap; the weights do not feel it (it
cancels in each row) but a held denominator would, so each row's eta is measured by a calibration circuit (the
Hadamard test of q_i with itself, which reads 1 on a perfect device) and the row sum is divided by eta^4.

Every <q_i|k_j> of every head and block of one e-SNLI test pair is read by a Hadamard test on 5 qubits: the state
(|0>|q_i> + |1>|k_j>) / sqrt 2 (q, k normalised; one head is 16 complex modes = 4 qubits, plus the ancilla) is
prepared directly, the ancilla gets an H and is measured: P(0) - P(1) = Re<q|k>; with -i on the |1> branch it
gives Im<q|k>. The norms |q| |k| are put back classically. The rest of the compiled pass (compile_hw) is exact.
Three readings are compared: exact, shots only (ideal circuits), and shots + the device's noise model
(qiskit_ibm_runtime.fake_provider snapshot, qiskit-aer). A uniform loss of visibility cancels in the attention
weights, which are normalised over each row.
"""
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import fake_provider

from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

args = sys.argv[1:]
opt = lambda k, v: args[args.index(k) + 1] if k in args else v
tag = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))][0]
idx, shots, bname = int(opt("--index", 15)), int(opt("--shots", 10000)), opt("--backend", "FakeTorino")
ncal = int(opt("--cal-random", 0))

model, d = load(tag)
backend = getattr(fake_provider, bname)()
noisy_sim, ideal_sim = AerSimulator.from_backend(backend), AerSimulator()
stats = dict(circuits=0, two_qubit=[], depth=[])
present = []
offline = []                                     # --offline: per block, circuits and CZ counts
calib = []                                       # per block: the fitted slope and offset of --cal-random


class Device:
    """The same circuits on an IBM device, one SamplerV2 job per block, through the hardware ledger of Seeing Why
    (src/ibm_jobs.py: a job is never paid for twice, a lost wait is fetched by its id, the quota is checked
    before sending, every job recorded), kept here in ibm_jobs/. Dry run (no --submit): each job is transpiled for the
    device and written to the ledger but not sent, and the pass goes on with ideal-simulator counts so the next block
    can be built (its circuits then differ from the real run's; their number and cost do not)."""

    def __init__(self, backend_name, keys, account, submit, purpose, pilot=0):
        import ibm_jobs                                                       # src/ibm_jobs.py; ledger in ibm_jobs/
        self.ij, self.backend_name, self.keys, self.account = ibm_jobs, backend_name, keys, account
        self.submit, self.purpose, self.folders, self.pilot = submit, purpose, [], pilot

    def run(self, circs, where, block):
        notes = [f"block {block} head {h} q{i} k{j} {('Re', 'Im', 'calibration')[part]}" if part < 3 else
                 f"block {block} random calibration {i} ideal {scale:+.6f}" for h, i, j, part, scale in where]
        # the ledger names a circuit by its gates and real parameters: StatePreparation's complex amplitudes become
        # u and cx gates first (deterministic, optimisation 0; the ledger then transpiles for the device)
        circs = transpile(circs, basis_gates=["u", "cx"], optimization_level=0)
        if "--offline" in args:                  # no account, no ledger: the --backend noise model, CZ counts kept
            t = transpile(circs, backend, optimization_level=3, seed_transpiler=0)
            cz = [c.count_ops().get("cz", 0) for c in t]
            offline.append(dict(block=block, circuits=len(t), cz_median=float(np.median(cz)), cz_max=int(max(cz)),
                                cz_calibration=[cz[n] for n, w in enumerate(where) if w[3] == 3][:5]))
            print(f"offline block {block}: {len(t)} circuits, CZ median {np.median(cz):.0f}, max {max(cz)}; "
                  f"calibration CZ {offline[-1]['cz_calibration']}", flush=True)
            res = noisy_sim.run(t, shots=shots, seed_simulator=block).result()
            return res
        if self.pilot:                                                        # the first circuits only, then stop
            from qiskit.quantum_info import Statevector
            circs, notes = circs[:self.pilot], notes[:self.pilot]
            counts, folder = self.ij.sampler_counts(circs, self.backend_name, shots, self.keys, self.account,
                                                    f"{self.purpose}: pilot, first {self.pilot} circuits of block 0",
                                                    inputs=[f"results/{tag}.pt"], circuit_notes=notes,
                                                    dry_run=not self.submit, repeat=1)
            exact = [Statevector(c.remove_final_measurements(inplace=False)).probabilities([4]) for c in circs]
            exact = np.array([p[0] - p[1] for p in exact])
            if counts is not None:
                meas = np.array([(c.get("0", 0) - c.get("1", 0)) / shots for c in counts])
                rec = json.loads((pathlib.Path(folder) / "record.json").read_text())
                print(f"pilot: {len(circs)} circuits x {shots} shots, {rec.get('quantum_seconds')} quantum s "
                      f"({rec.get('quantum_seconds', 0) / len(circs):.2f} s a circuit); P0 - P1 measured against "
                      f"ideal: corr {np.corrcoef(meas, exact)[0, 1]:.3f}, slope {np.polyfit(exact, meas, 1)[0]:.3f} "
                      f"(the visibility), max |diff| {np.abs(meas - exact).max():.3f}")
            else:
                print(f"pilot dry run: {len(circs)} circuits x {shots} shots written to {folder}, not sent")
            sys.exit()
        counts, folder = self.ij.sampler_counts(circs, self.backend_name, shots, self.keys, self.account,
                                                f"{self.purpose}, block {block}", inputs=[f"results/{tag}.pt"],
                                                circuit_notes=notes, dry_run=not self.submit)
        self.folders.append(str(folder))
        if counts is None:                     # dry run: go on with ideal counts (--noisy-dry: the noise model's)
            sim = noisy_sim if "--noisy-dry" in args else ideal_sim
            t = transpile(circs, backend if sim is noisy_sim else sim, optimization_level=3, seed_transpiler=0)
            res = sim.run(t, shots=shots, seed_simulator=block).result()
            counts = [res.get_counts(n) for n in range(len(circs))]
        return type("Counts", (), {"get_counts": lambda _, n: counts[n]})()


def circuit(q, k, imag):
    psi = np.concatenate([q, (-1j if imag else 1) * k]) / np.sqrt(2)          # ancilla = top qubit
    qc = QuantumCircuit(5, 1)
    qc.append(StatePreparation(psi), range(5))
    qc.h(4)
    qc.measure(4, 0)
    return qc


def make_overlap(sim, record, rows=None):
    """rows: a list that receives, per block, the measured row sums over the exact ones, [H, L] (calibrated)."""
    def overlap(qi, kj, block):
        q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()                  # [H, L, e]
        jobs, where = [], []
        for h in range(q.shape[0]):
            for i in present:
                for j in present:
                    nq, nk = np.linalg.norm(q[h, i]), np.linalg.norm(k[h, j])
                    if nq == 0 or nk == 0:
                        continue
                    for part in (0, 1):
                        jobs.append(circuit(q[h, i] / nq, k[h, j] / nk, part == 1))
                        where.append((h, i, j, part, nq * nk))
                if rows is not None and not ncal and np.linalg.norm(q[h, i]) > 0:   # visibility of row i
                    jobs.append(circuit(q[h, i] / np.linalg.norm(q[h, i]), q[h, i] / np.linalg.norm(q[h, i]), False))
                    where.append((h, i, i, 2, 1.0))
        if rows is not None and ncal:                     # same-depth calibrations: random pairs, known overlap
            # each pair its own calibration states (the ledger refuses a circuit already sent); pair 4667's job
            # was drawn with default_rng(1000 + block), its circuits and ideal overlaps are in the ledger
            rng = np.random.default_rng(1000 + block if idx == 4667 else [idx, block])
            for n in range(ncal):
                a, b = rng.standard_normal((2, 2, q.shape[-1]))
                a = (a[0] + 1j * a[1]) / np.linalg.norm(a[0] + 1j * a[1])
                r = b[0] + 1j * b[1]
                r = r - np.vdot(a, r) * a
                r /= np.linalg.norm(r)
                th = np.arccos(rng.uniform(-1, 1))                           # Re<a|b> spread over [-1, 1]
                bb = np.cos(th) * a + np.sin(th) * r
                jobs.append(circuit(a, bb, False))
                where.append((0, n, n, 3, float(np.vdot(a, bb).real)))
        if isinstance(sim, Device):                                          # an IBM device, through the ledger
            res = sim.run(jobs, where, block)
        else:
            t = transpile(jobs, backend if sim is noisy_sim else sim, optimization_level=3, seed_transpiler=0)
            if record:
                stats["circuits"] += len(t)
                stats["two_qubit"] += [c.count_ops().get("cz", 0) + c.count_ops().get("ecr", 0) for c in t]
                stats["depth"] += [c.depth() for c in t]
            res = sim.run(t, shots=shots, seed_simulator=block).result()
        L = q.shape[1]
        re, im = torch.zeros(1, q.shape[0], L, L, dtype=torch.float64), torch.zeros(1, q.shape[0], L, L,
                                                                                     dtype=torch.float64)
        eta = np.ones((q.shape[0], L))
        cal = []
        for n, (h, i, j, part, scale) in enumerate(where):
            c = res.get_counts(n)
            p = (c.get("0", 0) - c.get("1", 0)) / shots
            if part == 3:
                cal.append((scale, p))
            elif part == 2:
                eta[h, i] = max(p * scale, 1e-3)
            else:
                (im if part else re)[0, h, i, j] = p * scale
        if cal:                                          # one slope and one readout offset for the block
            x, y = np.array(cal).T
            slope, off = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
            calib.append(dict(block=block, slope=float(slope), offset=float(off), n=len(x),
                              resid_sd=float(np.std(y - slope * x - off))))
            for n, (h, i, j, part, scale) in enumerate(where):
                if part < 2:
                    (im if part else re)[0, h, i, j] -= off * scale
            eta[:] = slope
        if rows is not None and "--eta" in args:                            # a visibility measured elsewhere
            eta[:] = float(opt("--eta", 1))
        if rows is not None:
            power = 2 if model.attn == "born2" else 1
            ov = torch.einsum("hie,hje->hij", torch.tensor(q).conj(), torch.tensor(k))
            s_ex = (ov.real ** 2 + ov.imag ** 2) ** power
            s_hw = (re[0] ** 2 + im[0] ** 2) ** power / torch.tensor(eta)[:, :, None] ** (2 * power)
            keep = torch.zeros(L, dtype=torch.bool)
            keep[present] = True
            num, den = (s_hw * keep).sum(-1), (s_ex * keep).sum(-1)
            rows.append(torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den)))
        return re, im
    return overlap


def batch(n):
    """The first n test pairs, noisy circuits only: accuracy and agreement with the exact pass."""
    rows = []
    t0 = time.time()
    for i in range(n):
        ids = torch.tensor(d["x_test"][i:i + 1])
        present[:] = list(np.where(ids[0].numpy() != 0)[0]) + ([ids.shape[1]] if model.mean_channel else [])
        with torch.no_grad():
            pe, _ = compiled_probs(model, ids)
            pn, _ = compiled_probs(model, ids, overlap=make_overlap(noisy_sim, True))
        rows.append(dict(index=i, gold=int(d["y_test"][i]), exact=int(pe.argmax()), noisy=int(pn.argmax()),
                         p_diff=float((pn - pe).abs().max())))
        print(f"{i}: gold {rows[-1]['gold']} exact {rows[-1]['exact']} noisy {rows[-1]['noisy']} "
              f"|dP| {rows[-1]['p_diff']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    g, e, nz = (np.array([r[k] for r in rows]) for k in ("gold", "exact", "noisy"))
    res = dict(tag=tag, backend=bname, shots=shots, pairs=n, acc_exact=float((e == g).mean()),
               acc_noisy=float((nz == g).mean()), agree=float((nz == e).mean()),
               p_diff_median=float(np.median([r["p_diff"] for r in rows])),
               p_diff_max=float(max(r["p_diff"] for r in rows)), circuits=stats["circuits"],
               two_qubit_median=float(np.median(stats["two_qubit"])), seconds=time.time() - t0, rows=rows)
    with open(f"results/{tag}_hwatt_{bname}_{shots}_n{n}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


class HwFrozen(Frozen):
    """Frozen whose recorded attention row sums (the 4-d denominators) are scaled to the measured ones."""

    def __init__(self, rows):
        super().__init__()
        self.rows, self.blk = rows, 0

    def __call__(self, d):
        if not self.replay and d.dim() == 4:
            d = d * self.rows[self.blk][None, :, :, None].to(d.dtype)
            self.blk += 1
        return super().__call__(d)


def hw_evidence(ids, target, rows):
    """evidence.evidence with the denominators of HwFrozen."""
    x0, f = _score_fn(model, ids)
    x0 = x0.detach()
    fz = HwFrozen(rows)
    with torch.no_grad():
        full = f(x0, fz).gather(1, target[:, None])[:, 0]
        zero = f(torch.zeros_like(x0), fz.start_replay()).gather(1, target[:, None])[:, 0]
    total = torch.zeros_like(x0)
    for t, w in zip(*nodes01(math.ceil(model.degree() / 2))):
        x = (float(t) * x0).requires_grad_(True)
        out = f(x, fz.start_replay()).gather(1, target[:, None]).sum()
        (g,) = torch.autograd.grad(out, x)
        total += float(w) * g
    return (x0 * total).detach(), full, zero


def evidence_batch(n):
    rows_out, t0 = [], time.time()
    for i in range(n):
        ids = torch.tensor(d["x_test"][i:i + 1])
        present[:] = list(np.where(ids[0].numpy() != 0)[0]) + ([ids.shape[1]] if model.mean_channel else [])
        rows = []
        with torch.no_grad():
            pe, _ = compiled_probs(model, ids)
            pn, _ = compiled_probs(model, ids, overlap=make_overlap(noisy_sim, True, rows))
        target = pe.argmax(1)
        ev_ex, fe, ze = evidence(model, ids, target)
        ev_hw, fh, zh = hw_evidence(ids, target, rows)
        tok = np.where(ids[0].numpy() != 0)[0]
        a, b = ev_ex.detach()[0, tok].numpy(), ev_hw[0, tok].numpy()
        g = d["gt_test"][i][tok]
        k = int(g.sum()) or 4
        top = lambda e: set(np.argsort(-e)[:k])
        rows_out.append(dict(index=i, corr=float(np.corrcoef(a, b)[0, 1]), topk_agree=len(top(a) & top(b)) / k,
                             prec_exact=float(g[list(top(a))].mean()) if g.any() else None,
                             prec_hw=float(g[list(top(b))].mean()) if g.any() else None,
                             sum_hw=float(b.sum()), f_hw=float(fh - zh), p_exact=float(pe.max()),
                             p_hw=float(pn[0, target[0]]), eta=[float(r.mean()) for r in rows]))
        r = rows_out[-1]
        print(f"{i}: corr {r['corr']:.4f} top-k {r['topk_agree']:.2f} human {r['prec_exact']} / {r['prec_hw']} "
              f"sum {r['sum_hw']:.4f} = {r['f_hw']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    pe_ = [r["prec_exact"] for r in rows_out if r["prec_exact"] is not None]
    ph_ = [r["prec_hw"] for r in rows_out if r["prec_hw"] is not None]
    res = dict(tag=tag, backend=bname, shots=shots, pairs=n,
               corr_median=float(np.median([r["corr"] for r in rows_out])),
               corr_min=float(min(r["corr"] for r in rows_out)),
               topk_agree_mean=float(np.mean([r["topk_agree"] for r in rows_out])),
               human_prec_exact=float(np.mean(pe_)), human_prec_hw=float(np.mean(ph_)),
               completeness_gap_max=float(max(abs(r["sum_hw"] - r["f_hw"]) for r in rows_out)),
               seconds=time.time() - t0, rows=rows_out)
    with open(f"results/{tag}_hwev_{bname}_{shots}_n{n}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if "--device" in args:
    # one test pair on an IBM device (--device ibm_kingston --account <keys line> [--submit]); without --submit a dry
    # run. Answer and hybrid evidence against the exact pass, as evidence_batch does under the noise model.
    dev = Device(opt("--device", "ibm_kingston"), opt("--keys", None),
                 int(opt("--account", 1)), "--submit" in args,
                 f"Reading Why: Born-p2-ap attention of e-SNLI test pair {idx} by Hadamard tests",
                 pilot=int(opt("--pilot", 0)))
    ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
    if "--delete" in args:                         # the deletion test on the device: one word's amplitude set to 0
        dpos = int(opt("--delete", 0))
        dev.purpose += f", word {dpos} ({d['words'][int(ids[0, dpos])]}) deleted"
        ids[0, dpos] = 0
    present[:] = list(np.where(ids[0].numpy() != 0)[0]) + ([ids.shape[1]] if model.mean_channel else [])
    rows = None if "--no-evidence" in args else []   # no row calibrations without the hybrid evidence
    with torch.no_grad():
        pe, _ = compiled_probs(model, ids)
        ph, _ = compiled_probs(model, ids, overlap=make_overlap(dev, True, rows))
    if rows is None:
        out = dict(tag=tag, index=idx, deleted=int(opt("--delete", -1)), gold=d["classes"][int(d["y_test"][idx])],
                   device=dev.backend_name, submitted=dev.submit, shots=shots, ledger=dev.folders,
                   words=[d["words"][t] for t in ids[0].tolist() if t],
                   exact=dict(P=pe[0].tolist(), answer=d["classes"][int(pe.argmax())]),
                   device_run=dict(P=ph[0].tolist(), answer=d["classes"][int(ph.argmax())]),
                   p_diff_max=float((ph - pe).abs().max()))
        print(f"{idx} {' '.join(out['words'])}: exact {out['exact']['answer']} {np.round(pe[0].numpy(), 3).tolist()}  "
              f"{'device' if dev.submit else 'dry run (ideal counts)'} {out['device_run']['answer']} "
              f"{np.round(ph[0].numpy(), 3).tolist()}  |dP| {out['p_diff_max']:.3f}")
        with open(f"results/{tag}_device_{dev.backend_name}_{shots}_i{idx}_del{opt('--delete', 'none')}"
                  f"{'' if dev.submit else '_dry'}.json", "w") as f:
            json.dump(out, f, indent=1)
        sys.exit()
    target = pe.argmax(1)
    ev_ex, _, _ = evidence(model, ids, target)
    ev_hw, fh, zh = hw_evidence(ids, target, rows)
    tok = np.where(ids[0].numpy() != 0)[0]
    a, b = ev_ex.detach()[0, tok].numpy(), ev_hw[0, tok].numpy()
    words = [d["words"][t] for t in ids[0][tok].tolist()]
    out = dict(tag=tag, index=idx, gold=d["classes"][int(d["y_test"][idx])], device=dev.backend_name,
               submitted=dev.submit, shots=shots, ledger=dev.folders, words=words,
               exact=dict(P=pe[0].tolist(), answer=d["classes"][int(pe.argmax())], evidence=a.tolist()),
               device_run=dict(P=ph[0].tolist(), answer=d["classes"][int(ph.argmax())], evidence=b.tolist(),
                               eta=[r.tolist() for r in rows], evidence_sum=float(b.sum()), f_diff=float(fh - zh),
                               calibration=calib, offline=offline),
               evidence_corr=float(np.corrcoef(a, b)[0, 1]), p_diff_max=float((ph - pe).abs().max()))
    print(f"{idx} ({out['gold']}): exact {out['exact']['answer']} {np.round(pe[0].numpy(), 3).tolist()}  "
          f"{'device' if dev.submit else 'dry run (ideal counts)'} {out['device_run']['answer']} "
          f"{np.round(ph[0].numpy(), 3).tolist()}  |dP| {out['p_diff_max']:.3f}  evidence corr {out['evidence_corr']:.3f}")
    print("  evidence exact / device: " + ", ".join(f"{w} {x:+.3f}/{y:+.3f}" for w, x, y in zip(words, a, b)))
    eta_tag = (f"_eta{opt('--eta', '')}" if "--eta" in args else "") + (f"_cal{ncal}" if ncal else "") + (f"_offline_{bname}" if "--offline" in args else "")
    with open(f"results/{tag}_device_{dev.backend_name}_{shots}_i{idx}{eta_tag}{'' if dev.submit else '_dry'}.json", "w") as f:
        json.dump(out, f, indent=1)
    sys.exit()
if "--evidence" in args:
    evidence_batch(int(opt("--evidence", 20)))
    sys.exit()
if "--n" in args:
    batch(int(opt("--n", 50)))
    sys.exit()
ids = torch.tensor(d["x_test"][idx:idx + 1])
present[:] = list(np.where(ids[0].numpy() != 0)[0]) + ([ids.shape[1]] if model.mean_channel else [])
out = {}
with torch.no_grad():
    exact_trace, ideal_trace, noisy_trace = [], [], []
    p_exact, _ = compiled_probs(model, ids, trace=exact_trace)
    t0 = time.time()
    p_ideal, _ = compiled_probs(model, ids, trace=ideal_trace, overlap=make_overlap(ideal_sim, False))
    p_noisy, _ = compiled_probs(model, ids, trace=noisy_trace, overlap=make_overlap(noisy_sim, True))
    secs = time.time() - t0
classes = d["classes"]
att = lambda tr: [v[0][:, present][:, :, present].numpy() for n, v in tr if n == "attention"]
ae, ai, an = att(exact_trace), att(ideal_trace), att(noisy_trace)
for name, p, a in (("shots only", p_ideal, ai), ("shots + noise", p_noisy, an)):
    diffs = [float(np.abs(x - y).max()) for x, y in zip(a, ae)]
    corr = [float(np.corrcoef(x.ravel(), y.ravel())[0, 1]) for x, y in zip(a, ae)]
    out[name] = dict(P=p[0].tolist(), answer=classes[int(p.argmax())], attention_max_abs_diff=diffs,
                     attention_corr=corr, P_max_abs_diff=float((p - p_exact).abs().max()))
out["exact"] = dict(P=p_exact[0].tolist(), answer=classes[int(p_exact.argmax())])
out.update(tag=tag, index=idx, gold=classes[int(d["y_test"][idx])], backend=bname, shots=shots,
           circuits=stats["circuits"], two_qubit_median=float(np.median(stats["two_qubit"])),
           depth_median=float(np.median(stats["depth"])), seconds=secs)
with open(f"results/{tag}_hwatt_{bname}_{shots}.json", "w") as f:
    json.dump(out, f, indent=1)
torch.save(dict(exact=ae, ideal=ai, noisy=an, present=present), f"results/{tag}_hwatt_{bname}_{shots}.pt")
print(json.dumps(out, indent=1))
