"""Reading Why in a few lines: the answer, each word's Born evidence, and the answer with a word deleted.

    python examples/quickstart.py          (from the repository's top folder; a CPU is enough)

The same code is shown in docs/run-locally.md. examples/explain.py does the same for any pair from the command line.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"))
import reading_why as rw    # noqa: E402

m = rw.load("born")                    # the Born Transformer; or "softmax", or the tag of a run in results/
r = rw.explain(m, "A woman working long hours.", "A woman is sleeping.")
print(r["answer"], r["probs"])         # contradiction [0.019 0.006 0.975]
for w, e, a in zip(r["words"], r["evidence"], r["attention"]):
    print(f"{w:>10s}  evidence {e:+.4f}  attention {a:.3f}")
print(r["evidence"].sum(), r["explained"])            # equal (0.643164): the evidence sums to P_g - b_g

ids, words = rw.without(r, [r["words"].index("sleeping")])   # delete words in place, as the paper does
print(rw.explain(m, ids=ids, toks=words)["answer"])          # entailment
