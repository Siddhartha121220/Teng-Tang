"""Compare the fast shortlist suggester (settings from inference.N_CANDS) against the exact full-vocabulary one.

For sampled sentences it types a context and then 0, 1 and 2 letters of the real next word,
and reports latency per keystroke plus how often both versions give the same suggestions.

    python check_speedup.py [n_sentences]
"""
import random
import statistics
import sys
import time

import torch

import inference as inf


def cases(sentences, n, seed=0):
    rng = random.Random(seed)
    pool = [s for s in sentences if len(s) >= 2]
    for s in rng.sample(pool, min(n, len(pool))):
        i = rng.randrange(1, len(s))
        ctx, target = " ".join(s[:i]), s[i]
        typed = [ctx + " ", ctx + " " + target[:1]]
        if len(target) > 2:
            typed.append(ctx + " " + target[:2])
        yield target, typed


def main(n):
    print("torch threads:", torch.get_num_threads())
    for lang in inf.MODEL_DIRS:
        sizes = (inf.N_CANDS[lang],)
        base = inf._load_model(lang)
        lm = base.lm

        def make(n_cands):
            scorer = inf.LMScorer(lm.model, base.vocab, lm.tok, lm.eos_id, lm.device)
            return inf.Suggester(base.vocab, base.ng, scorer, base.alpha, base.normalize, n_cands)

        fast = {size: make(size) for size in sizes}
        exact = make(None)
        stats = {size: dict(times=[], same1=0, same3=0, hit=0) for size in sizes}
        hit_exact = total = 0

        for target, typed in cases(_meta_sentences(lang), n):
            for text in typed:
                e = exact.suggest(text)
                total += 1
                hit_exact += target in e
                for size, sug in fast.items():
                    t = time.perf_counter()
                    f = sug.suggest(text)
                    st = stats[size]
                    st["times"].append(time.perf_counter() - t)
                    st["same1"] += f[:1] == e[:1]
                    st["same3"] += set(f) == set(e)
                    st["hit"] += target in f

        print(f"\n== {lang} ({len(base.vocab)} words, {total} keystrokes) ==")
        print(f"exact (full vocab): target in top-3 {hit_exact / total:.0%}")
        for size, st in stats.items():
            print(f"n_cands={size:<4} median {statistics.median(st['times']) * 1000:5.0f} ms  "
                  f"max {max(st['times']) * 1000:5.0f} ms  "
                  f"same top-1 {st['same1'] / total:4.0%}  same top-3 {st['same3'] / total:4.0%}  "
                  f"target in top-3 {st['hit'] / total:4.0%}")
    print("\n(sentences come from the training data, so compare the columns, not the absolute hit rate)")


def _meta_sentences(lang):
    import json
    with open(inf.MODEL_DIRS[lang] / f"{lang}_meta.json", encoding="utf-8") as f:
        return json.load(f)["sentences"]


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
