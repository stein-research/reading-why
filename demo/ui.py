"""The interactive demo (ipywidgets). card() takes an explained pair and returns HTML, so it also works without
widgets; demo() wires it to controls.

    import ui; ui.demo()
"""
import html
import itertools

import ipywidgets as W
import matplotlib.colors as mc
import numpy as np
from IPython.display import HTML, display

import reading_why as rw

INK, MUTED, ACCENT, LINE = "#2b2f36", "#8a919a", "#d9480f", "#c9ced6"
# Text on the page background inherits the notebook's colour (light or dark theme); INK only on our own fills.
# MUTED is a mid grey that reads on white and on Colab's dark grey alike.
# Seeing Why's evidence colours: red for the answer, blue against it
EVIDENCE = mc.LinearSegmentedColormap.from_list(
    "evidence", ["#08306b", "#1f78d1", "#9cc9f5", "#f7f7f7", "#f9a77e", "#e8431c", "#7f0d0d"])
ATTENTION = mc.LinearSegmentedColormap.from_list("attention", ["#ffffff", "#fbe3c4", "#f08c3c", "#7a2a06"])
LABEL_FILL = dict(entailment="#d6ecd2", neutral="#dcdcf2", contradiction="#f9dcdf")
_MODELS = {}


def model(name):
    if name not in _MODELS:
        _MODELS[name] = rw.load(name)
    return _MODELS[name]


def _ink(rgb):
    r, g, b = rgb[:3]
    return "#ffffff" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else INK


def _chips(words, values, cmap, vmax, signed, marked=None, title=""):
    out = [f'<div style="color:{MUTED};font-size:12px;margin:8px 0 3px">{title}</div><div style="line-height:1.6">']
    for i, (w, v) in enumerate(zip(words, values)):
        if w == "<sep>":
            out.append(f'<span style="color:{MUTED};margin:0 8px">|</span>')
            continue
        x = (v / vmax + 1) / 2 if signed else v / vmax
        rgb = cmap(float(np.clip(x, 0, 1)))
        under = "border-bottom:3px solid #f2c200;" if marked and i < len(marked) and marked[i] else ""
        out.append(f'<span title="{v:+.4f}" style="display:inline-block;background:{mc.to_hex(rgb)};color:{_ink(rgb)};'
                   f'padding:3px 6px;margin:2px 1px;border-radius:4px;{under}font-size:15px;line-height:1.3">'
                   f'{html.escape(w)}</span>')
    return " ".join(out) + "</div>"                       # spaces: the chips wrap in a narrow window


def _bars(p, title):
    rows = []
    for k, lab in enumerate(rw.LABELS):
        top = k == int(np.argmax(p))
        rows.append(
            f'<div style="display:flex;align-items:center;margin:2px 0"><span style="width:105px;font-size:13px;'
            f'color:inherit;{"font-weight:600" if top else ""}">{lab}</span><span style="display:inline-block;'
            f'height:13px;width:{max(1, 220 * float(p[k])):.0f}px;background:{ACCENT if top else "#adb5bd"};'
            f'border-radius:2px"></span><span style="margin-left:6px;font-size:12px;color:{MUTED}">'
            f'{float(p[k]):.3f}</span></div>')
    return f'<div style="font-size:12px;color:{MUTED};margin-bottom:3px">{title}</div>' + "".join(rows)


def card(r, marked=None, gold=None, after=None, deleted=None):
    """One explained pair as HTML: the answer, the evidence per word, the attention per word, and (after) the answer
    once the words in `deleted` (a string, or a list of them) are deleted."""
    e, a = r["evidence"], r["attention"]
    v = max(float(np.abs(e).max()), 1e-12)
    name = dict(born="Born Transformer", softmax="softmax transformer", poly="Poly-sim-born")[r["model"]]
    exact = (f"the words' evidence sums to P &minus; b = {r['explained']:+.4f} (difference {r['gap']:.0e})"
             if r["model"] != "softmax" else
             f"integrated gradients of the logit; sums to {e.sum():+.3f} against {r['explained']:+.3f} (not exact)")
    head = (f'<div style="font-size:16px;color:inherit;margin-bottom:6px">{name}: <span style="color:{INK};background:'
            f'{LABEL_FILL[r["answer"]]};padding:2px 8px;border-radius:4px;font-weight:600">{r["answer"]}</span>'
            + (f' <span style="color:{MUTED};font-size:13px">(annotators: {gold})</span>' if gold else "") + "</div>")
    body = _chips(r["words"], e, EVIDENCE, v, True, marked,
                  f"Born evidence per word (BEW) for <b>{r['answer']}</b>: red for it, blue against it; "
                  f"{exact}" if r["model"] != "softmax" else
                  f"evidence (integrated gradients) for <b>{r['answer']}</b>: red for it, blue against it; {exact}")
    body += _chips(r["words"], a, ATTENTION, max(float(a.max()), 1e-12), False, marked,
                   "attention each word receives (mean over blocks, heads and words)")
    if marked:
        body += f'<div style="color:{MUTED};font-size:12px;margin-top:4px">yellow underline: words the annotators ' \
                f'highlighted</div>'
    side = _bars(r["probs"], "class probabilities")
    if after is not None:
        gone = [deleted] if isinstance(deleted, str) else list(deleted)
        side += "<div style='height:10px'></div>" + _bars(
            after["probs"], "without " + ", ".join(f'"{html.escape(w)}"' for w in gone))
    return (f'<div style="display:flex;gap:28px;flex-wrap:wrap;font-family:sans-serif"><div style="flex:1 1 420px;'
            f'min-width:0">{head}{body}</div><div style="flex:0 0 330px">{side}</div></div>')


def show(name, premise, hypothesis):
    """Without widgets: explain one pair with one model and display it."""
    display(HTML(card(rw.explain(model(name), premise, hypothesis))))


def demo(start=0):
    """Step through the e-SNLI test pairs (or type your own), choose the model, see the answer and why; delete a word
    to test it. prev / next take the labels in turn (entailment, neutral, contradiction, ...) in test order, or only
    the chosen label; random draws from the whole test split (or that label)."""
    pairs = rw.test_pairs()
    by = {lab: [i for i, p in enumerate(pairs) if p[2] == lab] for lab in rw.LABELS}
    turns = [i for trio in itertools.zip_longest(*by.values()) for i in trio if i is not None]   # labels in turn
    rng = np.random.default_rng()
    label = W.ToggleButtons(options=[("all", "all")] + [(lab, lab) for lab in rw.LABELS], value="all",
                            style={"button_width": "100px"})
    prev, nxt, rand = (W.Button(description=d, layout=W.Layout(width="90px")) for d in ("◀ prev", "next ▶", "random"))
    index = W.BoundedIntText(value=start, min=0, max=len(pairs) - 1, description="test pair",
                             layout=W.Layout(width="200px"))
    prem = W.Text(description="premise", layout=W.Layout(width="760px"))
    hyp = W.Text(description="hypothesis", layout=W.Layout(width="760px"))
    which = W.ToggleButtons(options=[("Born Transformer", "born"), ("softmax", "softmax")], value="born")
    go = W.Button(description="Explain", button_style="warning")
    words = W.HBox(layout=W.Layout(flex_flow="row wrap"))
    out = W.Output()
    state = {}

    def order():
        return turns if label.value == "all" else by[label.value]

    def step(d):
        o = order()
        i = index.value
        if i in o:
            k = (o.index(i) + d) % len(o)
        else:                                             # from a pair outside the order: the nearest one that way
            later = [j for j in o if (j > i if d > 0 else j < i)]
            k = o.index(min(later) if d > 0 else max(later)) if later else (0 if d > 0 else len(o) - 1)
        index.value = o[k]

    def fill(_=None):
        p = pairs[index.value]
        prem.value, hyp.value = p[0], p[1]

    def run(_=None, deleted=None):
        """deleted: the word positions to delete (a list), or None to explain the pair as it is."""
        out.clear_output()
        m = model(which.value)
        with out:
            if not prem.value.strip() or not hyp.value.strip():
                print("Type a premise and a hypothesis.")
                return
            r = rw.explain(m, prem.value, hyp.value)
            p = pairs[index.value]
            p = p if (prem.value, hyp.value) == p[:2] else None      # a typed pair has no annotators
            marked = rw.marked_mask(p) if p else None
            after = None
            if deleted:
                ids, toks = rw.without(r, deleted)
                after = rw.explain(m, ids=ids, toks=toks)
            state["r"] = r
            display(HTML(card(r, marked, p[2] if p else None, after,
                              [r["words"][i] for i in deleted] if after else None)))
        chosen = set(deleted or [])
        toggles = []
        for i, w in enumerate(state["r"]["words"]):
            if w == "<sep>":
                continue
            t = W.ToggleButton(description=w, value=i in chosen, layout=W.Layout(width="auto"),
                               button_style="warning" if i in chosen else "", tooltip=f"select '{w}' to delete")
            t.pos = i
            t.observe(lambda c, t=t: setattr(t, "button_style", "warning" if c["new"] else ""), names="value")
            toggles.append(t)
        delete = W.Button(description="delete selected", button_style="danger", layout=W.Layout(width="auto"))
        delete.on_click(lambda _: run(deleted=sorted(t.pos for t in toggles if t.value)))
        clear = W.Button(description="clear", layout=W.Layout(width="auto"))
        clear.on_click(lambda _: run())
        words.children = [W.Label("pick words, then delete:")] + toggles + [delete, clear]

    prev.on_click(lambda _: step(-1))
    nxt.on_click(lambda _: step(+1))
    rand.on_click(lambda _: setattr(index, "value", int(rng.choice(order()))))
    index.observe(lambda c: (fill(), run()), names="value")
    which.observe(lambda c: run(), names="value")
    go.on_click(run)
    fill()
    display(W.VBox([W.HBox([prev, nxt, rand, index]), label, prem, hyp, W.HBox([which, go]), out, words]))
    run()
