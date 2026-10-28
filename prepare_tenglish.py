"""
prepare_tenglish.py
Turn WhatsApp chat exports (.txt, "Export chat -> Without media") into
tenglish_train.csv with columns [category, sentence], the format the
autocomplete notebook expects.

Usage:
    python prepare_tenglish.py chats/*.txt --names siddhartha sid karthik anuj --out tenglish_train.csv

    --names      names/nicknames to strip (yours and your friends'); case-insensitive
    --min-seed   how many Telugu seed words a line needs to count as Tenglish (default 1)
    --no-filter  keep every cleaned line (use to inspect what the filter throws away)
"""
import argparse, csv, glob, re, sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------- message line formats
# Android:  12/03/25, 9:41 pm - Karthik: em chestunnav ra
# iOS:      [12/03/25, 9:41:23 PM] Karthik: em chestunnav ra
ANDROID = re.compile(r"^\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP]\.?\s?[mM]\.?)?\s+-\s+([^:]{1,60}):\s?(.*)$")
IOS     = re.compile(r"^\[\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP]\.?\s?[mM]\.?)?\]\s+([^:]{1,60}):\s?(.*)$")
# WhatsApp Web copy-paste:  [9:41 pm, 12/3/2025] Karthik: em chestunnav ra   (time first, then date)
WEB     = re.compile(r"^\[\d{1,2}:\d{2}(?::\d{2})?\s*(?:[apAP]\.?\s?[mM]\.?)?,\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\]\s+([^:]{1,60}):\s?(.*)$")
DATE_START = re.compile(r"^\[?(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4},|\d{1,2}:\d{2}[^\]]{0,12}\])")   # any new entry, incl. system notices

# ---------------------------------------------------------------- junk to drop
SKIP_EXACT = {
    "<media omitted>", "image omitted", "video omitted", "audio omitted", "sticker omitted",
    "gif omitted", "document omitted", "this message was deleted", "you deleted this message",
    "null", "missed voice call", "missed video call", "waiting for this message",
    "<view once message omitted>", "view once message omitted",
}
SKIP_CONTAINS = ("omitted>", ".vcf", "location:", "poll:", "live location shared",
                 "end-to-end encrypted", "security code changed")

URL    = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL  = re.compile(r"\S+@\S+\.\S+")
MENTION = re.compile(r"@\S+")
DIGITS = re.compile(r"\d+")
TELUGU_SCRIPT = re.compile(r"[\u0C00-\u0C7F]")
TAMIL_SCRIPT  = re.compile(r"[\u0B80-\u0BFF]")
REPEAT = re.compile(r"([a-z])\1{2,}")                  # raaaa -> raa, sooo -> soo
SPLIT  = re.compile(r"[.!?\n]+")                        # break long messages into clauses

# ---------------------------------------------------------------- Tenglish seed words
# Common romanized Telugu function/verb words. Extend this after looking at the
# "most common words" report printed at the end.
TELUGU_SEED = set("""
ra raa ri re ro enti entra entraa emi em emaina emo ela elaa ela undi undhi unnav unnava unnaru unna
ledu ledhu leda ledha ledura kadu kaadu kada kadha kadaa avunu avuna aithe ayithe ayindi ayyindi
aipoyindi aypoyindi poyindi vachindi vachadu vachava vastha vasthunna vastunna vasta vastava velthunna
veltha veldam veltunna vellu velli po poyi podam cheppu cheppa cheppava chepthe cheppanu chesthunna
chestunna chesthuna chesa chesanu chesava chey cheyyi cheyali chuddam chudu choodu chusa chusava
chusina tinnava tinna tindam tinu padukunna nenu nuvvu nuvu meeru manam memu vadu vaadu vallu adi idi
akkada ikkada ekkada enduku endhuku evaru evadu enni entha antha anni inka inkaa ippudu ipudu appudu
eppudu epudu repu ninna ivala ivvala konchem koncham chala chaala kuda kooda mari sare sarle
ante antav baaga bagundi baagundi bagane bagaledu pani emaindi abba ayyo orey arey oye babu bava mama
oka okka rendu gurinchi kosam tho lo ki ni ga ani kani ayite tarvata ippude appude vundi vunnav unnadu naaku naku neeku neku maaku meeku vaadiki dhani deeniki
nen adhi adhe untadhi anta anthe nuv iga nee malla endhi
asalu asal unde chestha ippud etla endhuk kadhu unte atla
edho ayyindhi motham kaadhu ochindhi poi aame malli anukunna niku
""".split())

# Words that are also ordinary English / abbreviations ("lo and behold", "re: meeting", "mama"),
# so one of them alone is NOT enough evidence that a line is Tenglish. A line needs one strong
# word, or two weak ones.
WEAK = set("re ro ri ga ni lo ki po mama babu bava ante pani mari oka adi idi tho ata anna egh".split())
STRONG = TELUGU_SEED - WEAK

def clean_message(text, name_pat):
    t = text.strip().lower()
    t = t.replace("<this message was edited>", "").replace("\u200e", "").strip()
    if not t or t in SKIP_EXACT or any(s in t for s in SKIP_CONTAINS):
        return []
    if TELUGU_SCRIPT.search(t) or TAMIL_SCRIPT.search(t):     # native script: not romanized code-mix
        return []
    if re.search(r"\d{5,}", t.replace(" ", "")):             # phone numbers, OTPs, account nos: drop whole message
        return []
    t = URL.sub(" ", t); t = EMAIL.sub(" ", t); t = MENTION.sub(" ", t); t = DIGITS.sub(" ", t)
    if name_pat is not None:
        t = name_pat.sub(" ", t)
    t = REPEAT.sub(r"\1\1", t)
    out = []
    for clause in SPLIT.split(t):
        words = tokenize_words(clause)
        if words:
            out.append(words)
    return out

def tokenize_words(text):
    """Identical to the notebook's tokenizer, so the CSV is already in its final form."""
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", str(text).lower())
    return re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", text)

def read_messages(path):
    """Yield message bodies, joining multi-line messages and skipping system notices."""
    current = None
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.replace("\u202f", " ").replace("\u200e", "").strip()
        m = ANDROID.match(line) or IOS.match(line) or WEB.match(line)
        if m:
            if current is not None:
                yield current
            current = m.group(2)
        elif DATE_START.match(line):          # system notice (joined, left, changed icon, ...)
            if current is not None:
                yield current
            current = None
        elif current is not None and line:    # continuation of a multi-line message
            current += "\n" + line
    if current is not None:
        yield current

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--names", nargs="*", default=[])
    ap.add_argument("--out", default="tenglish_train.csv")
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=15)
    ap.add_argument("--min-seed", type=int, default=1)
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()

    name_pat = None
    if args.names:
        alts = "|".join(sorted((re.escape(n.lower()) for n in args.names), key=len, reverse=True))
        name_pat = re.compile(rf"\b(?:{alts})(?:'?s)?\b")

    files = sorted({f for pat in args.files for f in (glob.glob(pat) or [pat])})   # expands *.txt on Windows too
    rows, seen, stats, dropped = [], set(), Counter(), []
    for i, f in enumerate(files, 1):
        category = f"chat_{i}"                 # anonymous id instead of the chat/friend name
        for msg in read_messages(f):
            stats["messages"] += 1
            for words in clean_message(msg, name_pat):
                if not (args.min_words <= len(words) <= args.max_words):
                    stats["dropped_length"] += 1; continue
                if not args.no_filter:
                    strong = sum(w in STRONG for w in words)
                    weak = sum(w in WEAK for w in words)
                    if strong < args.min_seed and weak < 2:
                        stats["dropped_not_tenglish"] += 1
                        dropped.append(" ".join(words)); continue
                key = " ".join(words)
                if key in seen:
                    stats["dropped_duplicate"] += 1; continue
                seen.add(key)
                rows.append((category, key))
        print(f"{f} -> {category}", file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["category", "sentence"]); w.writerows(rows)

    dpath = Path(args.out).with_name("dropped_not_tenglish.txt")
    dpath.write_text("\n".join(dropped), encoding="utf-8")
    dcounts = Counter(w for d in dropped for w in d.split())
    print(f"\nDropped-as-not-Tenglish lines saved to {dpath} (private, delete it before sharing anything).")
    print("Most common words in those dropped lines. Any ROMANIZED TELUGU word here is missing from")
    print("TELUGU_SEED: add it to the list in this script and re-run to recover those lines:")
    print(dcounts.most_common(50))

    counts = Counter(w for _, s in rows for w in s.split())
    print(f"\nMessages read: {stats['messages']}")
    print(f"Dropped: length={stats['dropped_length']}  not-tenglish={stats['dropped_not_tenglish']}  "
          f"duplicate={stats['dropped_duplicate']}")
    print(f"Kept sentences: {len(rows)}  -> {args.out}")
    print(f"Per chat: {dict(Counter(c for c, _ in rows))}")
    print(f"Unique words: {len(counts)} | seen only once: {sum(1 for c in counts.values() if c == 1)}")
    print("Most common words (check these for spelling variants and leftover names):")
    print(counts.most_common(40))

if __name__ == "__main__":
    main()
