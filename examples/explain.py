"""Explain one premise and hypothesis from the command line: the answer, each word's Born evidence (BEW) and the
attention it receives, and the answer again with words deleted. Runs on a CPU; the first run downloads the e-SNLI
test split only if you pass --test.

    python examples/explain.py "A woman working long hours." "A woman is sleeping."
    python examples/explain.py "A woman working long hours." "A woman is sleeping." --delete sleeping
    python examples/explain.py --test 117 --model softmax
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo"))
import reading_why as rw    # noqa: E402

p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("premise", nargs="?")
p.add_argument("hypothesis", nargs="?")
p.add_argument("--test", type=int, help="explain e-SNLI test pair N instead (with the annotators' highlights)")
p.add_argument("--model", default="born", help="born (default), softmax, or the tag of a run in results/")
p.add_argument("--delete", nargs="+", default=[], metavar="WORD", help="words to delete (every occurrence)")
args = p.parse_args()

marked = None
if args.test is not None:
    pair = rw.test_pairs()[args.test]
    premise, hypothesis = pair[0], pair[1]
    marked = rw.marked_mask(pair)
    print(f"test pair {args.test} (annotators: {pair[2]})")
elif args.premise and args.hypothesis:
    premise, hypothesis = args.premise, args.hypothesis
else:
    p.error("give a premise and a hypothesis, or --test N")

m = rw.load(args.model)
r = rw.explain(m, premise, hypothesis)
print(f"premise:    {premise}\nhypothesis: {hypothesis}")
print(f"answer: {r['answer']}   " + "   ".join(f"{c} {q:.3f}" for c, q in zip(rw.LABELS, r["probs"])))
print(f"\n{'word':>12s}  {'evidence':>9s}  {'attention':>9s}" + ("  highlighted" if marked else ""))
for i, (w, e, a) in enumerate(zip(r["words"], r["evidence"], r["attention"])):
    hl = "  *" if marked and i < len(marked) and marked[i] else ""
    print(f"{w:>12s}  {e:+9.4f}  {a:9.3f}{hl}")
what = "P - b, exact" if r["model"] != "softmax" else "not exact for the softmax model"
print(f"\nsum of evidence {r['evidence'].sum():+.6f}, explained {r['explained']:+.6f} ({what})")

if args.delete:
    pos = [i for i, w in enumerate(r["words"]) if w in {d.lower() for d in args.delete}]
    if not pos:
        sys.exit(f"none of {args.delete} is in the pair")
    ids, words = rw.without(r, pos)
    a = rw.explain(m, ids=ids, toks=words)
    print(f"without {', '.join(repr(r['words'][i]) for i in pos)}: {a['answer']}   "
          + "   ".join(f"{c} {q:.3f}" for c, q in zip(rw.LABELS, a["probs"])))
