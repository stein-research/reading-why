"""The shot-noise floor of the Hadamard-test evidence on the two device pairs (referee round 5, 2026-11-12): synthetic
readings slope * ideal + binomial noise at 2,000 shots, calibrated with the true slope, row sums and hybrid evidence as
the paper builds them, 60 draws; also with a Gaussian of sd 0.022 per circuit added (a per-circuit excess that
reproduces the device's row log-sd). And device_power.py's uniform/random overlap nulls as they were (absent slots
taking a share of each row) beside correctly normalised ones.

python device_floor.py   -> results/device_floor.json   (CPU, about ten minutes)
"""
import json, math, re
import numpy as np, torch
from compile_hw import compiled_probs, load
from evidence import _score_fn, evidence, nodes01
from models import Frozen

OUT = "results/device_floor.json"
model, d = load("esnli_Born-ap_b2_s0_e20")
SLOPES = {3763: (0.7668, 0.6737), 3765: (0.8146, 0.7947)}
DRAWS = 60
rng = np.random.default_rng(0)


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
for idx in (3763, 3765):
    ids = torch.tensor(d["x_test"][idx:idx + 1])
    tok = np.where(ids[0].numpy() != 0)[0]
    present = list(tok) + [ids.shape[1]]
    with torch.no_grad():
        pe = compiled_probs(model, ids)[0]
    target = pe.argmax(1)
    a = evidence(model, ids, target)[0].detach()[0, tok].numpy()

    def make(extra):
        rows = []

        def overlap(qi, kj, block):
            q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy()
            H, L = q.shape[0], q.shape[1]
            slope = SLOPES[idx][block]
            exact = np.einsum("hie,hje->hij", q.conj(), k)
            nn = np.linalg.norm(q, axis=-1)[:, :, None] * np.linalg.norm(k, axis=-1)[:, None, :]
            re_, im_ = torch.zeros(1, H, L, L, dtype=torch.float64), torch.zeros(1, H, L, L, dtype=torch.float64)
            for h in range(H):
                for i in present:
                    for j in present:
                        if nn[h, i, j] == 0:
                            continue
                        for imag in (False, True):
                            ideal = (exact[h, i, j].imag if imag else exact[h, i, j].real) / nn[h, i, j]
                            p = np.clip(slope * ideal, -1, 1)
                            v = 2 * rng.binomial(2000, (1 + p) / 2) / 2000 - 1 + extra * rng.standard_normal()
                            (im_ if imag else re_)[0, h, i, j] = v * nn[h, i, j]
            s_ex = torch.tensor(np.abs(exact) ** 2)
            s_hw = (re_[0] ** 2 + im_[0] ** 2) / slope ** 2
            keep = torch.zeros(L, dtype=torch.bool); keep[present] = True
            num, den = (s_hw * keep).sum(-1), (s_ex * keep).sum(-1)
            rows.append(torch.where(den > 0, num / den.clamp(min=1e-300), torch.ones_like(den)))
            return re_, im_
        return overlap, rows

    r = {}
    for name, extra in (("shot_only", 0.0), ("shot_plus_0.022_gauss", 0.022)):
        corrs, l1s, rowsd, same = [], [], [], 0
        for _ in range(DRAWS):
            overlap, rows = make(extra)
            with torch.no_grad():
                p = compiled_probs(model, ids, overlap=overlap)[0][0]
            same += int(p.argmax() == target[0])
            b = hw_evidence(ids, target, rows, tok)
            corrs.append(float(np.corrcoef(a, b)[0, 1])); l1s.append(float(np.abs(a - b).sum() / np.abs(a).sum()))
            rowsd.append(float(np.mean([np.log(rw[:, present].numpy().ravel()).std() for rw in rows])))
        r[name] = dict(corr_median=float(np.median(corrs)), corr_p5=float(np.percentile(corrs, 5)), corr_p95=float(np.percentile(corrs, 95)),
                       l1_median=float(np.median(l1s)), row_log_sd_median=float(np.median(rowsd)), answer_same=same, draws=DRAWS)
        print(idx, name, r[name], flush=True)
    res[str(idx)] = r

# the paper's uniform / random overlap functions (device_power.py) on all three device inputs
def uniform_paper(qi, kj, block):
    ov = (qi.conj() * kj).sum(-1)
    return torch.ones_like(ov.real), torch.zeros_like(ov.real)


def random_paper(qi, kj, block):
    ov = (qi.conj() * kj).sum(-1)
    return torch.tensor(rng.standard_normal(ov.shape)), torch.tensor(rng.standard_normal(ov.shape))


res["answers"] = {}
for name, idx, dele in (("3763", 3763, None), ("3765", 3765, None), ("3763 without aliens", 3763, 5)):
    ids = torch.tensor(d["x_test"][idx:idx + 1]).clone()
    if dele is not None:
        ids[0, dele] = 0
    tok = np.where(ids[0].numpy() != 0)[0]
    present = list(tok) + [ids.shape[1]]

    def uniform_ok(qi, kj, block):
        H, L = qi.shape[1], qi.shape[2]
        re_ = torch.zeros(1, H, L, L, dtype=torch.float64)
        for i in present:
            for j in present:
                re_[0, :, i, j] = 1.0
        return re_, torch.zeros_like(re_)

    def random_ok(qi, kj, block):
        H, L = qi.shape[1], qi.shape[2]
        re_ = torch.zeros(1, H, L, L, dtype=torch.float64); im_ = torch.zeros_like(re_)
        for i in present:
            for j in present:
                re_[0, :, i, j] = torch.tensor(rng.standard_normal(H)); im_[0, :, i, j] = torch.tensor(rng.standard_normal(H))
        return re_, im_

    def attn_sum(fn):
        tr = []
        with torch.no_grad():
            p = compiled_probs(model, ids, overlap=fn, trace=tr)[0][0]
        w = [v for n, v in tr if n == "attention"][0]
        return p, float(w[0, 0, present[0]].sum())

    with torch.no_grad():
        pe = compiled_probs(model, ids)[0][0]
        pu_paper, su_paper = attn_sum(uniform_paper)
        pu_ok, su_ok = attn_sum(uniform_ok)
        rp = sum(int(compiled_probs(model, ids, overlap=random_paper)[0][0].argmax() == pe.argmax()) for _ in range(100))
        ro = sum(int(compiled_probs(model, ids, overlap=random_ok)[0][0].argmax() == pe.argmax()) for _ in range(100))
    res["answers"][name] = dict(exact_p=pe.tolist(), uniform_paper_p=pu_paper.tolist(), uniform_paper_row_sum=su_paper,
                                uniform_normalised_p=pu_ok.tolist(), uniform_normalised_row_sum=su_ok,
                                random_paper_same_of_100=rp, random_normalised_same_of_100=ro)
    print(name, res["answers"][name], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
