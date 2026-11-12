"""Trivial baselines and checks for the e-SNLI evaluation (referee round 5, 2026-11-12), on faithfulness.py's 1,000
test pairs: flip@1 and agreement of rankings that need no model (random; hypothesis tokens first; rarest hypothesis
content word first; last token first), against the model's own evidence; where the evidence's top token lies
(separator, punctuation, hypothesis, premise) and flip@1 split by it; the evidence's sum gap and node convergence; and
the Born model's dependence on a common scale of the amplitudes (x1.5, and sqrt(23/22), the deletion's renormalisation
at the median length).

python esnli_baselines.py [tags...]   -> results/esnli_baselines.json   (GPU, a few minutes per model)
"""
import json, sys, time
import numpy as np, torch
import torch.nn.functional as F
from compile_hw import load
from evidence import evidence

dev = "cuda"
OUT = "results/esnli_baselines.json"
tags = sys.argv[1:] or ["esnli_Born-ap_b2_s0_e20", "esnli_Born-ap_b2_s1_e20", "esnli_Born-ap_b2_s2_e20",
                        "esnli_Softmax_b2_s0_e20", "esnli_Poly-sim-born_b2_s0_e20"]
res = {}
d = None
for tag in tags:
    t0 = time.time()
    model, d = load(tag)
    model = model.to(dev)
    words = d["words"]
    pick = np.random.default_rng(0).choice(len(d["x_test"]), 1000, replace=False)
    ids = torch.tensor(d["x_test"][pick]).to(dev)
    gt = torch.tensor(d["gt_test"][pick]).to(dev)
    prob = (lambda x: model(x)) if model.family == "B" else (lambda x: F.softmax(model(x), -1))
    with torch.no_grad():
        p0 = prob(ids)
    target = p0.argmax(1)
    present = ids != 0
    n_tok = present.sum(1)
    r = {"acc_1000": float((target == torch.tensor(d["y_test"][pick]).to(dev)).float().mean())}
    # --- exactness
    evs, gaps = [], []
    for i in range(0, 1000, 100):
        ev, full, zero = evidence(model, ids[i:i + 100], target[i:i + 100])
        evs.append(ev.detach()); gaps.append(((ev.sum(1) - (full - zero)).abs() / (full - zero).abs().clamp(min=1e-300)).detach())
    ev, gap = torch.cat(evs), torch.cat(gaps)
    r["gap_median"], r["gap_max"] = float(gap.median()), float(gap.max())
    if model.family == "B":
        e30 = evidence(model, ids[:10], target[:10], n_nodes=30)[0].detach()
        e400 = evidence(model, ids[:10], target[:10], n_nodes=400)[0].detach()
        e100 = evidence(model, ids[:10], target[:10], n_nodes=100)[0].detach()
        r["nodes_30_vs_400"], r["nodes_100_vs_400"] = float((e30 - e400).abs().max()), float((e100 - e400).abs().max())
        # scale dependence of the model: common factor on every amplitude
        with torch.no_grad():
            amp = model.amplitudes(ids[:200])
            pa = model.numerators(ids[:200], amp); pa = pa[0] / pa[1][:, None] + model.share
            pb = model.numerators(ids[:200], amp * 1.5); pb = pb[0] / pb[1][:, None] + model.share
            pc = model.numerators(ids[:200], amp * np.sqrt(23 / 22)); pc = pc[0] / pc[1][:, None] + model.share
        r["scale_1.5_maxdP"], r["scale_1.5_answer_changed"] = float((pa - pb).abs().max()), float((pa.argmax(1) != pb.argmax(1)).float().mean())
        r["scale_sqrt23over22_maxdP"] = float((pa - pc).abs().max())
    # --- rankings
    L = ids.shape[1]
    sep_pos = (ids == 2).float().argmax(1)
    pos = torch.arange(L, device=dev)[None].expand_as(ids)
    in_hyp = present & (pos > sep_pos[:, None])
    in_prem = present & (pos < sep_pos[:, None])
    is_punct = torch.tensor([not any(ch.isalnum() for ch in w) for w in words], device=dev)[ids] & present
    is_sep = ids == 2
    content = present & ~is_punct & ~is_sep
    big = torch.finfo(torch.float64).max
    rng = torch.Generator(device=dev).manual_seed(1)

    def rnd():
        return torch.rand(ids.shape, generator=rng, device=dev, dtype=torch.float64)

    def order_of(score):
        return torch.where(present, score, torch.full_like(score, -big)).argsort(1, descending=True)

    def flip1(order):
        out = ids.clone(); out.scatter_(1, order[:, :1], 0)
        with torch.no_grad():
            return float((prob(out).argmax(1) != target).float().mean())

    def agree(order):
        g = gt & present
        k = g.sum(1)
        rank = torch.empty_like(order).scatter_(1, order, torch.arange(L, device=dev).expand_as(order))
        topk = (rank < k[:, None]) & present
        has = k > 0
        return float((((topk & g).sum(1).float() / k.clamp(min=1).float())[has]).mean())

    rank_id = ids.double()  # larger id = rarer word (vocabulary sorted by frequency)
    scores = {
        "evidence": lambda: ev,
        "random": lambda: rnd(),
        "hyp_random": lambda: rnd() + in_hyp.double() * 10,
        "hyp_content_random": lambda: rnd() + in_hyp.double() * 10 + content.double() * 5,
        "hyp_rarest_first": lambda: rank_id + in_hyp.double() * 1e6 + content.double() * 1e5,
        "rarest_first": lambda: rank_id + content.double() * 1e6,
        "last_tokens_first": lambda: pos.double(),
    }
    r["rank"] = {}
    for name, s in scores.items():
        reps = 5 if "random" in name else 1
        fl, ag = [], []
        for _ in range(reps):
            o = order_of(s())
            fl.append(flip1(o)); ag.append(agree(o))
        r["rank"][name] = dict(flip1=float(np.mean(fl)), agreement=float(np.mean(ag)))
    # where the evidence's top word is
    top = order_of(ev)[:, 0]
    r["top1_is_sep"] = float(is_sep.gather(1, top[:, None]).float().mean())
    r["top1_is_punct"] = float(is_punct.gather(1, top[:, None]).float().mean())
    r["top1_in_hyp"] = float(in_hyp.gather(1, top[:, None]).float().mean())
    r["top1_in_prem"] = float(in_prem.gather(1, top[:, None]).float().mean())
    # flip@1 split by whether the top word is sep/punct
    o = order_of(ev)
    out = ids.clone(); out.scatter_(1, o[:, :1], 0)
    with torch.no_grad():
        fl = (prob(out).argmax(1) != target)
    sp = (is_sep | is_punct).gather(1, top[:, None])[:, 0]
    r["flip1_when_top_is_sep_or_punct"] = float(fl[sp].float().mean()) if sp.any() else None
    r["flip1_when_top_is_word"] = float(fl[~sp].float().mean())
    # evidence mass on the hypothesis vs premise (for the predicted class)
    r["evidence_share_hyp"] = float(((ev * in_hyp).sum(1) / ev.abs().sum(1)).mean())
    r["evidence_share_sep"] = float(((ev * is_sep).sum(1) / ev.abs().sum(1)).mean())
    r["seconds"] = time.time() - t0
    res[tag] = r
    print(tag, json.dumps(r, indent=1), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
