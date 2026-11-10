"""Every single-word deletion of the two device pairs (3763, 3765) in the three seeds of the Born Transformer, with
each pair's exact evidence (BEW): which deletions change the answer, and where the evidence ranks those words.

python deletion_words.py   # writes results/deletion_3763_3765_born_ap.json
"""
import json, sys, numpy as np, torch
sys.path.insert(0, ".")
import synth
from models import build
from evidence import evidence
d = synth.get("esnli"); V, C = len(d["words"]), len(d["classes"])
def load(tag):
    cfg = json.load(open(f"results/{tag}.json"))
    m = build(cfg["model"], V, C, blocks=cfg["blocks"], max_len=64)
    m.load_state_dict(torch.load(f"results/{tag}.pt", map_location="cpu")); return m.double().eval()
def P(m, ids):
    with torch.no_grad():
        o = m(ids)
        return (o if m.family == "B" else torch.log_softmax(o, -1).exp())[0].numpy()
out = {}
for tag in ["esnli_Born-ap_b2_s%d_e20" % s for s in range(3)]:
    m = load(tag)
    for idx in (3763, 3765):
        ids = torch.tensor(d["x_test"][idx:idx+1]).clone()
        pos = [p for p in range(ids.shape[1]) if ids[0, p]]
        words = [d["words"][int(ids[0, p])] for p in pos]
        p0 = P(m, ids); t = int(p0.argmax())
        ev = evidence(m, ids, torch.tensor([t]))[0][0]
        ev = [float(ev[p]) for p in pos] if len(ev) > len(pos) else [float(x) for x in ev]
        dels = []
        for p in pos:
            j = ids.clone(); j[0, p] = 0; pd = P(m, j)
            dels.append(dict(word=d["words"][int(ids[0, p])], pos=p, P=np.round(pd, 3).tolist(), answer=d["classes"][int(pd.argmax())]))
        out[f"{tag}|{idx}"] = dict(words=words, full=np.round(p0, 3).tolist(), answer=d["classes"][t], evidence=np.round(ev, 3).tolist(), deletions=dels)
        print(tag, idx, d["classes"][t], np.round(p0, 3).tolist())
        print("   ev", list(zip(words, np.round(ev, 3).tolist())))
        for x in dels: print("   -", x["word"], x["P"], x["answer"])
json.dump(out, open("results/deletion_3763_3765_born_ap.json", "w"), indent=1)
