"""Figure: P_g - b_g along the integration path a -> alpha a, with the denominators held (BEW) and free, on two test
pairs of the Born Transformer (seed 0, 20 epochs): a typical one (9574) and one whose free path starts far above
b_g (4322). Held, the path is a polynomial that starts at zero; free, it is flat below alpha ~ 1e-2 at the model's
answer for vanishing amplitudes, which is not b_g because the reference slot does not scale with alpha. Also prints,
over the tables' 1,000 pairs, the distance of that start from b_g. models.GUARD is set to 1e-300 (see
free_denominators.py: at 1e-30 the guard, not the model, sets the free path below alpha ~ 1e-7).

python fig_free_path.py [--out results/free_path] [--pairs 9574 4322]
"""
import glob
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch

import models
import synth
from evidence import _score_fn
from models import Frozen, build

models.GUARD = 1e-300

for f in glob.glob("/usr/share/texmf/fonts/opentype/public/lm/lmsans10-*.otf"):
    fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Latin Modern Sans", "DejaVu Sans"], "mathtext.fontset": "custom",
                     "mathtext.rm": "Latin Modern Sans", "mathtext.it": "Latin Modern Sans:italic",
                     "mathtext.bf": "Latin Modern Sans:bold", "pdf.fonttype": 3})
INK, MUTED, LINE, ACCENT, BLUE = "#2b2f36", "#495057", "#c9ced6", "#d9480f", "#1c7ed6"

args = sys.argv[1:]
out = args[args.index("--out") + 1] if "--out" in args else "results/free_path"
show = [int(s) for s in args[args.index("--pairs") + 1:args.index("--pairs") + 3]] if "--pairs" in args else [9574, 4322]
dev = "cuda" if torch.cuda.is_available() else "cpu"
d = synth.get("esnli")
pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)      # the tables' 1,000 pairs
IDS = torch.tensor(d["x_test"][pick]).to(dev)
tag = "esnli_Born-ap_b2_s0_e20"
cfg = json.load(open(f"results/{tag}.json"))
model = build(cfg["model"], len(d["words"]), len(d["classes"]), blocks=cfg["blocks"], max_len=IDS.shape[1]).to(dev)
model.load_state_dict(torch.load(f"results/{tag}.pt", map_location=dev))
model.double().eval()

alphas = np.concatenate([np.logspace(-6, -2, 33)[:-1], np.linspace(0.01, 1, 300)])
free, held = [], []
with torch.no_grad():
    for i in range(0, len(IDS), 100):
        ids = IDS[i:i + 100]
        tgt = model(ids).argmax(1)[:, None]
        x0, f = _score_fn(model, ids)
        fz = Frozen()
        f(x0, fz)                                              # records the denominators at the input
        free.append(torch.stack([f(a * x0, lambda v: v).gather(1, tgt)[:, 0] for a in alphas], 1))
        held.append(torch.stack([f(a * x0, fz.start_replay()).gather(1, tgt)[:, 0] for a in alphas], 1))
free, held = torch.cat(free).cpu().numpy(), torch.cat(held).cpu().numpy()

start = free[:, 0]                                             # alpha = 1e-6: on the flat part
print(json.dumps({"pairs": len(IDS), "free_start_minus_b_median_abs": float(np.median(np.abs(start))),
                  "free_start_minus_b_max_abs": float(np.max(np.abs(start))),
                  "free_flat_1e-6_to_1e-2_median_abs": float(np.median(np.abs(free[:, 32] - free[:, 0]))),
                  "held_at_0_max_abs": float(np.abs(held[:, 0]).max())}, indent=1))

fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.5), sharey=True)
for ax, idx in zip(axs, show):
    j = int(np.where(pick == idx)[0][0])
    ax.plot(alphas, free[j], color=ACCENT, lw=1.5, label="denominators free")
    ax.plot(alphas, held[j], color=BLUE, lw=1.5, label="held (BEW)")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(alphas[0], 1)
    ax.set_xlabel(r"$\alpha$ (path $\alpha a$)", color=INK)
    ax.set_title(f"pair {idx}", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(LINE)
axs[0].set_ylabel(r"$P_g - b_g$", color=INK)
axs[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03)
print(f"wrote {out}.png / .pdf")
