"""Order-free bag-of-embeddings baselines for e-SNLI (referee round 6, 2026-11-12), under the paper's protocol (Adam 2e-3,
cosine, batch 64, word dropout 0.1, gradient clip 1, 20 epochs, last epoch): a 64-d embedding, the mean over the
premise u and over the hypothesis v, no positions and no attention, then
  inter   [u, v, u*v, |u-v|] -> 300 ReLU -> 3
  concat  [u, v] -> 256 ReLU -> 256 ReLU -> 3
python bag_baseline.py inter 0 20   -> results/bag_inter_s0.json and .pt   (GPU, about 7 minutes)
"""
import math, sys, time, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import esnli
variant, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed); np.random.seed(seed)
d = esnli.load(); dev = "cuda"
V = len(d["words"])
class DAN(nn.Module):
    def __init__(s, variant, dim=64, hid=300):
        super().__init__()
        s.variant = variant; s.emb = nn.Embedding(V, dim)
        nin = {"concat": 2*dim, "inter": 4*dim}[variant]
        if variant == "concat": s.mlp = nn.Sequential(nn.Linear(nin, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3))
        else: s.mlp = nn.Sequential(nn.Linear(nin, hid), nn.ReLU(), nn.Linear(hid, 3))
    def forward(s, ids):
        e = s.emb(ids); pres = ids != 0
        sep = (ids == 2).int().argmax(1)                       # position of <sep> (0 if dropped -> all goes to hypothesis side)
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        hassep = (ids == 2).any(1)
        mp = (pres & (pos < sep[:, None]) & hassep[:, None]).float()
        mh = (pres & ((pos > sep[:, None]) | ~hassep[:, None]) & (ids != 2)).float()
        u = (e * mp[..., None]).sum(1) / mp.sum(1, keepdim=True).clamp(min=1)
        v = (e * mh[..., None]).sum(1) / mh.sum(1, keepdim=True).clamp(min=1)
        f = torch.cat((u, v), 1) if s.variant == "concat" else torch.cat((u, v, u * v, (u - v).abs()), 1)
        return s.mlp(f)
m = DAN(variant).to(dev)
ne = sum(p.numel() for n, p in m.named_parameters() if not n.startswith("emb"))
print(variant, "non-embedding params", ne, "total", sum(p.numel() for p in m.parameters()), flush=True)
xtr, ytr = torch.tensor(d["x_train"]), torch.tensor(d["y_train"])
def acc(split):
    x, y = torch.tensor(d[f"x_{split}"], device=dev), torch.tensor(d[f"y_{split}"], device=dev)
    m.eval()
    with torch.no_grad(): a = (m(x).argmax(1) == y).float().mean().item()
    m.train(); return a
opt = torch.optim.Adam(m.parameters(), lr=2e-3)
steps = epochs * math.ceil(len(xtr) / 64)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
t0 = time.time()
hist = []
for ep in range(epochs):
    perm = torch.randperm(len(xtr))
    for i in range(0, len(xtr), 64):
        idx = perm[i:i+64]; ids = xtr[idx].clone()
        cut = (torch.rand(ids.shape) < 0.1) & (ids != 0)
        keep = (ids != 0) & ~cut; cut[keep.sum(1) == 0] = False; ids[cut] = 0
        loss = F.cross_entropy(m(ids.to(dev)), ytr[idx].to(dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    hist.append(dict(epoch=ep + 1, dev=acc("dev"), test=acc("test"))); print(f"epoch {ep+1} dev {hist[-1]['dev']:.4f} test {hist[-1]['test']:.4f} ({time.time()-t0:.0f}s)", flush=True)
torch.save(m.state_dict(), f"results/bag_{variant}_s{seed}.pt")
json.dump(dict(variant=variant, seed=seed, epochs=epochs, non_embedding_params=ne, dev=hist[-1]["dev"], test=hist[-1]["test"], history=hist), open(f"results/bag_{variant}_s{seed}.json", "w"), indent=1)
