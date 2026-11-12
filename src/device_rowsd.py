"""Does independent per-circuit noise explain the device's spread of attention row sums? (referee round 6, 2026-11-12)
Pairs 3763 and 3765, Hadamard tests at 2,000 shots with each block's calibrated slope: synthetic readings with shot
noise plus a Gaussian of sd `extra` per circuit (in the readings' units: 0, 0.015, 0.022), 200 draws; the median row
log-sd of measured/exact row sums and the per-circuit error in ideal units, to set against results/device_errors.json
(per-circuit 0.035/0.039 and 0.035/0.040; row log-sd 0.085 and 0.064 in results/device_power.json).

python device_rowsd.py   -> results/device_rowsd.json   (CPU, a few minutes)
"""
import json, numpy as np, torch
from compile_hw import compiled_probs, load
model, d = load("esnli_Born-ap_b2_s0_e20")
SL = {3763: (0.7668, 0.6737), 3765: (0.8146, 0.7947)}
rng = np.random.default_rng(1)
OUT = {}
L0 = np.array([(d["x_test"][i] != 0).sum() for i in range(len(d["x_test"]))]); print("median tokens", np.median(L0), "n test", len(L0))
for idx in (3763, 3765):
    ids = torch.tensor(d["x_test"][idx:idx + 1]); tok = np.where(ids[0].numpy() != 0)[0]; present = list(tok) + [ids.shape[1]]
    for extra in (0.0, 0.0150, 0.022):
        sds, res = [], []
        for _ in range(200):
            rows = []
            def overlap(qi, kj, block):
                q, k = qi[0, :, :, 0].numpy(), kj[0, :, 0].numpy(); H, L = q.shape[:2]; s = SL[idx][block]
                ex = np.einsum("hie,hje->hij", q.conj(), k)
                nn = np.linalg.norm(q, axis=-1)[:, :, None] * np.linalg.norm(k, axis=-1)[:, None, :]
                P = np.ix_(range(H), present, present)
                re_, im_ = np.zeros((H, L, L)), np.zeros((H, L, L))
                for arr, part in ((re_, ex.real), (im_, ex.imag)):
                    ideal = part[P] / nn[P]
                    v = 2 * rng.binomial(2000, (1 + np.clip(s * ideal, -1, 1)) / 2) / 2000 - 1 + extra * rng.standard_normal(ideal.shape)
                    res.append((v / s - ideal).std())
                    arr[P] = v * nn[P]
                shw = (re_ ** 2 + im_ ** 2) / s ** 2
                rows.append((shw[P].sum(-1) / (np.abs(ex) ** 2)[P].sum(-1)))
                return torch.tensor(re_)[None], torch.tensor(im_)[None]
            with torch.no_grad(): compiled_probs(model, ids, overlap=overlap)
            sds.append(np.mean([np.log(r.ravel()).std() for r in rows]))
        OUT.setdefault(str(idx), {})[str(extra)] = dict(row_log_sd_median=float(np.median(sds)), row_log_sd_p5_p95=[float(np.percentile(sds, 5)), float(np.percentile(sds, 95))], draws=len(sds), per_circuit_sd_blocks=np.array(res).reshape(-1, 4).mean(0).tolist())
        print(idx, extra, "row log sd median", round(float(np.median(sds)), 4), "per-circuit sd (ideal units) blocks", np.round(np.array(res).reshape(-1, 4).mean(0), 4))
json.dump(OUT, open("results/device_rowsd.json", "w"), indent=1)
