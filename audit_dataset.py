#!/usr/bin/env python3
"""Audit tenglish_train.csv before uploading to Kaggle. Read-only: never edits the data."""
import argparse
import csv
import random
import re
from collections import Counter

PHONE_RE = re.compile(r"\d{5,}")
EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NAME_LIKE_RE = re.compile(r"^[A-Z][a-z]{2,}$")


def tokenize_words(text):
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", text.lower())
    return re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", text)


def edit_distance_1_clusters(words, min_count=2):
    """Group words that are edit-distance-1 apart (cheap: bucket by length+first letter first)."""
    by_bucket = {}
    for w in words:
        by_bucket.setdefault((len(w), w[0] if w else ""), []).append(w)

    def close(a, b):
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) == len(b):
            diffs = sum(1 for x, y in zip(a, b) if x != y)
            return diffs == 1
        s, l = (a, b) if len(a) < len(b) else (b, a)
        i = j = skipped = 0
        while i < len(s) and j < len(l):
            if s[i] != l[j]:
                skipped += 1
                if skipped > 1:
                    return False
                j += 1
            else:
                i += 1
                j += 1
        return True

    def shared_prefix(a, b, n=3):
        return a[:n] == b[:n]

    seen = set()
    clusters = []
    keys = list(by_bucket)
    for i, k1 in enumerate(keys):
        for k2 in keys[i:]:
            if abs(k1[0] - k2[0]) > 1:
                continue
            for a in by_bucket[k1]:
                for b in by_bucket[k2]:
                    if a == b or (a, b) in seen or (b, a) in seen:
                        continue
                    seen.add((a, b))
                    if shared_prefix(a, b) and close(a, b):
                        clusters.append((a, b))
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", nargs="?", default="tenglish_train.csv")
    ap.add_argument("--names", nargs="*", default=[], help="names already stripped by prepare_tenglish.py, to cross-check")
    ap.add_argument("--sample", type=int, default=40)
    args = ap.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"=== {args.csv_path}: {len(rows)} rows ===\n")

    # --- per-category balance ---
    cats = Counter(r["category"] for r in rows)
    print("Rows per category:")
    for c, n in sorted(cats.items(), key=lambda x: -x[1]):
        pct = n / len(rows) * 100
        flag = "  <-- over 50% of dataset" if pct > 50 else ""
        print(f"  {c}: {n} ({pct:.1f}%){flag}")
    print()

    # --- word-level stats ---
    all_words = []
    lengths = Counter()
    for r in rows:
        ws = tokenize_words(r["sentence"])
        all_words.extend(ws)
        lengths[len(ws)] += 1
    word_counts = Counter(all_words)
    print(f"Unique words: {len(word_counts)}")
    print(f"Words seen only once: {sum(1 for c in word_counts.values() if c == 1)} "
          f"({sum(1 for c in word_counts.values() if c == 1) / len(word_counts):.0%} of vocab)")
    print("Sentence length distribution (words):", dict(sorted(lengths.items())))
    print()

    print("Top 100 words:")
    for w, c in word_counts.most_common(100):
        print(f"  {w}: {c}")
    print()

    # --- spelling-variant candidates for NORMALIZE ---
    common_words = [w for w, c in word_counts.items() if c >= 2 and len(w) >= 4]
    clusters = edit_distance_1_clusters(common_words)
    print(f"Spelling-variant candidates (edit-distance-1, len>=4, count>=2): {len(clusters)} pairs")
    for a, b in sorted(clusters)[:60]:
        print(f"  {a!r} <-> {b!r}  (counts: {word_counts[a]}, {word_counts[b]})")
    if len(clusters) > 60:
        print(f"  ... and {len(clusters) - 60} more")
    print()

    # --- PII scan ---
    print("=== PII scan ===")
    phone_hits, email_hits, url_hits, name_hits = [], [], [], []
    for i, r in enumerate(rows):
        s = r["sentence"]
        if PHONE_RE.search(s):
            phone_hits.append((i, s))
        if EMAIL_RE.search(s):
            email_hits.append((i, s))
        if URL_RE.search(s):
            url_hits.append((i, s))
    print(f"Rows with 5+ digit runs (phone/OTP-like): {len(phone_hits)}")
    for i, s in phone_hits[:10]:
        print(f"  row {i}: {s}")
    print(f"Rows with emails: {len(email_hits)}")
    for i, s in email_hits[:10]:
        print(f"  row {i}: {s}")
    print(f"Rows with URLs: {len(url_hits)}")
    for i, s in url_hits[:10]:
        print(f"  row {i}: {s}")

    # capitalised name-like tokens (dead check once the pipeline lowercases everything, kept for --no-filter/raw input)
    cap_tokens = Counter()
    for r in rows:
        for tok in r["sentence"].split():
            if NAME_LIKE_RE.match(tok):
                cap_tokens[tok] += 1
    print(f"\nCapitalised name-like tokens still present: {len(cap_tokens)}")
    for w, c in cap_tokens.most_common(30):
        print(f"  {w}: {c}")

    # third-party names: prepare_tenglish.py only strips --names (you + direct chat partner);
    # names of OTHER people mentioned inside a conversation slip through lowercased. Flag words
    # sitting next to common Telugu name-honorifics as candidates.
    NAME_PARTICLES = {"gadu", "babu", "anna", "akka", "babai", "mava", "garu"}
    name_cand = Counter()
    for r in rows:
        words = r["sentence"].split()
        for i, w in enumerate(words):
            if w in NAME_PARTICLES:
                if i > 0 and words[i - 1] not in NAME_PARTICLES:
                    name_cand[words[i - 1]] += 1
                if i + 1 < len(words) and words[i + 1] not in NAME_PARTICLES:
                    name_cand[words[i + 1]] += 1
    print(f"\nPossible third-party names (word adjacent to {sorted(NAME_PARTICLES)}):")
    for w, c in name_cand.most_common(20):
        print(f"  {w}: {c}")

    if args.names:
        leftover = [n for n in args.names if n.lower() in word_counts]
        print(f"\n--names given: {args.names}")
        print(f"Still present in data despite --names: {leftover if leftover else 'none'}")
    print()

    # --- random sample for eyeballing ---
    print(f"=== {args.sample} random rows ===")
    random.seed(42)
    for r in random.sample(rows, min(args.sample, len(rows))):
        print(f"  [{r['category']}] {r['sentence']}")


if __name__ == "__main__":
    main()
