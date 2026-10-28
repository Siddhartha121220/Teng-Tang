"""meta.json -> Gboard personal-dictionary zip (Gboard > Settings > Dictionary > Personal dictionary > ⋮ > Import).

Format (from a known-working import file): header "# Gboard Dictionary version:1", then
"shortcut<TAB>word<TAB>locale" per line (shortcut first, per the LaTeX-Gboard-Dictionary maintainer);
empty shortcut = plain word, empty locale = all languages.
Gboard stores no frequencies or word pairs, so we drop words seen fewer than MIN_COUNT times (typos, one-off names).
"""
import json, sys, zipfile
from collections import Counter

meta_path, out_zip = sys.argv[1], sys.argv[2]
MIN_COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 2

counts = Counter(w for s in json.load(open(meta_path))["sentences"] for w in s)
words = sorted((w for w, c in counts.items() if c >= MIN_COUNT), key=lambda w: -counts[w])
text = "# Gboard Dictionary version:1\n" + "".join(f"\t{w}\t\n" for w in words)
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("dictionary.txt", text)
print(f"{out_zip}: {len(words)} words (seen >= {MIN_COUNT}x)")
