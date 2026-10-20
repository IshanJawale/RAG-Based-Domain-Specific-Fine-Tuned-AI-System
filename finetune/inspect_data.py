import json, sys, collections, re
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
D = Path(__file__).parent / "data"

def rd(n):
    p = D / n
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []

q, c, r = rd("questions.jsonl"), rd("contexts.jsonl"), rd("raw.jsonl")
print("questions", len(q), collections.Counter(x["kind"] for x in q))
print("contexts ", len(c), collections.Counter(x["kind"] for x in c))
print("raw      ", len(r), collections.Counter(x["kind"] for x in r))
print("dup qids q/c/r:", len(q) - len({x['qid'] for x in q}), len(c) - len({x['qid'] for x in c}), len(r) - len({x['qid'] for x in r}))
sq = [x["question"] for x in q if x["kind"] == "seed"]
wl = [len(s.split()) for s in sq]
print("seed q words avg/max:", sum(wl) / max(1, len(wl)), max(wl or [0]))
print("seed q starting 'In 20':", sum(s.startswith("In 20") for s in sq), "mention Rs.:", sum("Rs." in s for s in sq))
print("styles:", collections.Counter(x["style"] for x in q if x["kind"] == "seed"))
print("oos cats:", collections.Counter(x["style"] for x in q if x["kind"] == "oos"))
print("oos dup text:", len([x for x in q if x['kind']=='oos']) - len({x['question'] for x in q if x['kind']=='oos'}))
if c:
    print("seed_hit rate:", sum(x["seed_hit"] for x in c if x["kind"] == "seed") / max(1, sum(1 for x in c if x["kind"] == "seed")))
if r:
    ref = sum(x["answer"].strip().startswith("The available legal evidence is insufficient") for x in r)
    print("refusals:", ref, "/", len(r), "| empty answers:", sum(not x["answer"].strip() for x in r),
          "| route ok:", sum(bool(x.get("route")) for x in r))
    print("answers with [E#]:", sum(bool(re.search(r"\[E\d+\]", x["answer"])) for x in r))
    for x in r[:2]:
        print("---\nQ:", x["question"][:200], "\nA:", x["answer"][:500])
print("--- sample seed questions")
for s in sq[::max(1, len(sq)//8)][:8]:
    print("*", s[:220])
