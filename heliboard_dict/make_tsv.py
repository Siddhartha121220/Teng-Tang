"""meta.json sentences -> unigrams.tsv / bigrams.tsv for TenglishDictCompiler.

Scores are 1..255 on ONE log-probability scale (255 = probability 1):
  unigram  -> P(w)            = count(w) / N
  bigram   -> P(w | prev)     = (count(prev, w) - 0.75) / count(prev)   (same discount as the Python n-gram)
The dictionary format stores a bigram only as a step *above* the target word's unigram score,
so both must share a scale, otherwise context-predicted words get no boost.
"""
import json, math, sys
from collections import Counter

meta_path, top_n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9  # default: whole vocabulary
sents = json.load(open(meta_path))["sentences"]
uni, bi = Counter(), Counter()
for s in sents:
    uni.update(s)
    bi.update(zip(s, s[1:]))

N = sum(uni.values())
LOG_FLOOR = math.log(1 / (2 * N))  # probability that maps to score 0

def score(p):
    return max(1, min(255, round(255 * (1 - math.log(p) / LOG_FLOOR))))

top = [w for w, _ in uni.most_common(top_n)]
keep = set(top)
with open("unigrams.tsv", "w") as f:
    for w in top:
        f.write(f"{w}\t{score(uni[w] / N)}\n")
n_bi = 0
with open("bigrams.tsv", "w") as f:
    for (a, b), c in bi.items():
        if a in keep and b in keep:
            f.write(f"{a}\t{b}\t{score((c - 0.75) / uni[a])}\n")
            n_bi += 1
print(f"{len(top)} unigrams, {n_bi} bigrams")
