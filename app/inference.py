"""
inference.py — Local inference for the Tanglish / Tenglish DistilGPT2 models.

Both languages use the same ranking-hybrid architecture (N-gram + fine-tuned
DistilGPT2, blended by alpha) — see MODEL_DIRS below for where each language's
weights live. Models are loaded lazily and cached per language.
"""

import copy
import logging
import json
import os
import re
import threading
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model locations — one folder per language, each with its own meta json
# (tanglish_meta.json / tenglish_meta.json), produced by the training notebook.
# ---------------------------------------------------------------------------
_MODELS_ROOT = Path(__file__).parent / "models"
MODEL_DIRS = {
    "tanglish": _MODELS_ROOT / "tanglish",
    "tenglish": _MODELS_ROOT / "tenglish",
}

# How many n-gram candidates DistilGPT2 re-ranks per keystroke (see check_speedup.py).
# Tenglish's vocabulary is ~4x larger, so it needs a longer shortlist to keep accuracy.
N_CANDS = {"tanglish": 50, "tenglish": 200}
# While a word is being typed, at most this many extra prefix-matching words are scored
# on top of the context's shortlist (which is already scored and cached).
N_PREFIX_EXTRA = 50

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Default 4 = the i5-13420H's performance cores; more threads spill onto its slow
# efficiency cores and every suggestion gets slower. Override with TORCH_THREADS=n.
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))

_model_lock = threading.Lock()
_prefetcher = ThreadPoolExecutor(max_workers=1)

# language -> loaded Suggester (populated lazily by _load_model)
_suggesters: dict[str, "Suggester"] = {}


def tokenize_words(text, normalize=None):
    text = re.sub(r"(?<=[a-z0-9])\.(?=[a-z0-9])", "", str(text).lower())
    words = re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", text)
    if normalize:
        words = [normalize.get(w, w) for w in words]
    return words


BOS, EOS_W = "<s>", "</s>"


class NGram:
    def __init__(self, sents, vocab, discount=0.75):
        self.vocab = vocab
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self.w2i[EOS_W] = len(vocab)
        self.V = len(vocab)
        self.d = discount
        self.c1 = defaultdict(Counter)
        self.c2 = defaultdict(Counter)
        left = defaultdict(set)
        for s in sents:
            seq = [BOS, BOS] + s + [EOS_W]
            for i in range(2, len(seq)):
                self.c1[(seq[i - 1],)][seq[i]] += 1
                self.c2[(seq[i - 2], seq[i - 1])][seq[i]] += 1
                left[seq[i]].add(seq[i - 1])
        uni = np.full(self.V + 1, 0.1)
        for w, i in self.w2i.items():
            uni[i] += len(left.get(w, ()))
        self.uni = uni / uni.sum()

    def _interp(self, lower, counts):
        if not counts:
            return lower
        total = sum(counts.values())
        out = lower * (self.d * len(counts) / total)
        for w, n in counts.items():
            j = self.w2i.get(w)
            if j is not None:
                out[j] += max(n - self.d, 0.0) / total
        return out

    def probs(self, ctx):
        seq = [BOS, BOS] + list(ctx)
        p = self.uni.copy()
        p = self._interp(p, self.c1.get((seq[-1],)))
        p = self._interp(p, self.c2.get((seq[-2], seq[-1])))
        p = p[:self.V]
        return p / p.sum()


class LMScorer:
    def __init__(self, model, vocab, tok, eos_id, device, chunk=128):
        self.model, self.vocab, self.chunk = model.eval(), vocab, chunk
        self.tok, self.eos_id, self.device = tok, eos_id, device
        self.first = [tok.encode(w) for w in vocab]
        self.later = [tok.encode(" " + w) for w in vocab]
        n_tok = model.get_output_embeddings().weight.shape[0]
        boundary = []
        for i in range(min(n_tok, len(tok))):
            s = tok.decode([i])
            if i == eos_id or (s and (s[0].isspace() or not (s[0].isalnum() or s[0] == "'"))):
                boundary.append(i)
        self.boundary = torch.tensor(boundary, device=device)
        self.cache = {}  # ctx tuple -> {vocab index: raw log-score}

    def raw_scores(self, ctx, word_ids):
        """Unnormalised log-score of each vocab word in word_ids following ctx.
        Only words not already cached for this ctx are run through the model."""
        key = tuple(ctx)
        with _model_lock:  # requests and background prefetches share the cache and the CPU
            if key not in self.cache:
                if len(self.cache) > 5000:
                    self.cache.clear()  # ponytail: crude bound, swap for an LRU if memory matters
                self.cache[key] = {}
            known = self.cache[key]
            missing = [i for i in word_ids if i not in known]
            if missing:
                known.update(zip(missing, self._score(ctx, missing)))
            return np.array([known[i] for i in word_ids])

    def is_cached(self, ctx):
        return tuple(ctx) in self.cache

    @torch.no_grad()
    def _score(self, ctx, word_ids):
        prefix = [self.eos_id] + (self.tok.encode(" ".join(ctx)) if ctx else [])
        P = len(prefix)
        cands = self.later if ctx else self.first

        # The typed context is identical for every candidate, so run it through the
        # model once and reuse its attention cache, instead of once per candidate.
        out = self.model(input_ids=torch.tensor([prefix], device=self.device), use_cache=True)
        first_lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)  # predicts each candidate's 1st token
        prefix_cache = out.past_key_values

        order = sorted(range(len(word_ids)), key=lambda j: len(cands[word_ids[j]]))
        scores = np.zeros(len(word_ids))
        for a in range(0, len(order), self.chunk):
            pos = order[a:a + self.chunk]
            idx = [word_ids[j] for j in pos]
            n, L = len(idx), max(len(cands[i]) for i in idx)
            ids = torch.full((n, L), self.eos_id, dtype=torch.long)
            att = torch.zeros((n, P + L), dtype=torch.long)
            att[:, :P] = 1
            tgt = torch.zeros((n, L), dtype=torch.long)
            msk = torch.zeros((n, L))
            last = torch.zeros(n, dtype=torch.long)
            for r, i in enumerate(idx):
                c = cands[i]
                ids[r, :len(c)] = torch.tensor(c)
                att[r, P:P + len(c)] = 1
                tgt[r, :len(c)] = torch.tensor(c)
                msk[r, :len(c)] = 1
                last[r] = len(c) - 1
            ids, att, tgt, msk, last = (t.to(self.device) for t in (ids, att, tgt, msk, last))

            cache = copy.deepcopy(prefix_cache)
            cache.batch_repeat_interleave(n)
            # position j of the output predicts candidate token j + 1 (or the word boundary)
            lp = F.log_softmax(self.model(input_ids=ids, attention_mask=att,
                                          past_key_values=cache).logits.float(), dim=-1)
            rest_lp = lp[:, :L - 1, :].gather(2, tgt[:, 1:].unsqueeze(-1)).squeeze(-1)
            word_lp = first_lp[tgt[:, 0]] + (rest_lp * msk[:, 1:]).sum(1)
            end_lp = torch.logsumexp(lp[torch.arange(n, device=self.device), last][:, self.boundary], dim=-1)
            scores[pos] = (word_lp + end_lp).cpu().numpy()
        return scores


class Suggester:
    """Ranks words by alpha * P_lm + (1 - alpha) * P_ngram.

    Speed-up: DistilGPT2 only scores a shortlist of n_cands words picked by the
    (cheap) n-gram, instead of the whole vocabulary. Its probabilities are
    normalised over the n-gram's top n_cands words for the context, which hold
    almost all of the probability mass. While a word is being typed, only up to
    n_prefix_extra matching words beyond that shortlist are scored.
    n_cands=None scores the full vocabulary (the original, exact behaviour from
    the notebook).
    """

    def __init__(self, vocab, ngram, lm, alpha, normalize, n_cands=200, n_prefix_extra=N_PREFIX_EXTRA):
        self.vocab, self.ng, self.lm, self.alpha, self.normalize = vocab, ngram, lm, alpha, normalize
        self.n_cands, self.n_prefix_extra = n_cands, n_prefix_extra
        self.w2i = {w: i for i, w in enumerate(vocab)}
        self._starts = {}
        self._pending = None  # the running background prefetch, if any

    def prefix_mask(self, prefix):
        if prefix not in self._starts:
            self._starts[prefix] = np.array([w.startswith(prefix) for w in self.vocab])
        return self._starts[prefix]

    def _top(self, p, n):
        """Indices of the n highest entries of p (all of them if n is None)."""
        if n is None or n >= len(p):
            return np.argsort(-p)
        top = np.argpartition(-p, n)[:n]
        return top[np.argsort(-p[top])]

    def suggest(self, text, k=3):
        words = tokenize_words(text, self.normalize)
        typing_word = bool(text) and not text[-1].isspace() and bool(words)
        prefix = words[-1] if typing_word else ""
        ctx = words[:-1] if typing_word else words

        p_ng = self.ng.probs(ctx)
        base = self._top(p_ng, self.n_cands)
        log_z = np.logaddexp.reduce(self.lm.raw_scores(ctx, base))

        if prefix:
            mask = self.prefix_mask(prefix)
            if not mask.any():
                return [prefix]
            # matching words already on the context's (scored) shortlist, plus a few more
            extra = self._top(np.where(mask, p_ng, -1.0), self.n_prefix_extra)
            cands = np.union1d(base[mask[base]], extra[mask[extra]])
        else:
            cands = base

        p_lm = np.exp(self.lm.raw_scores(ctx, cands) - log_z)
        p = self.alpha * p_lm + (1 - self.alpha) * p_ng[cands]
        top = [self.vocab[cands[i]] for i in np.argsort(-p)[:k]]
        if prefix and top:
            self._prefetch(ctx + [top[0]])
        return top

    def _prefetch(self, ctx):
        """Score the most likely next context in the background, so the keystroke that
        finishes this word (space, or clicking the suggestion) is answered from the cache."""
        if self.lm.is_cached(ctx) or (self._pending is not None and not self._pending.done()):
            return
        self._pending = _prefetcher.submit(
            lambda: self.lm.raw_scores(ctx, self._top(self.ng.probs(ctx), self.n_cands)))


def _load_model(language: str) -> Suggester:
    if language not in MODEL_DIRS:
        raise ValueError(f"Unknown language {language!r}; expected one of {list(MODEL_DIRS)}")
    if language in _suggesters:
        return _suggesters[language]

    model_dir = MODEL_DIRS[language]
    if not model_dir.exists():
        raise RuntimeError(
            f"Model directory not found: {model_dir}\n"
            f"Extract the {language} model zip (from Kaggle output) into {model_dir}."
        )

    meta_path = model_dir / f"{language}_meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"{meta_path.name} not found in {model_dir}")

    logger.info("[%s] Loading tokenizer from %s ...", language, model_dir)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    logger.info("[%s] Loading model on %s ...", language, _device.upper())
    lm = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
    ).to(_device).eval()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    normalize = meta.get("normalize", {})
    vocab = sorted({w for s in meta["sentences"] for w in s})
    ngram = NGram(meta["sentences"], vocab, meta["discount"])
    lm_scorer = LMScorer(lm, vocab, tok, eos_id, _device)
    suggester = Suggester(vocab, ngram, lm_scorer, meta["alpha"], normalize, N_CANDS[language])

    _suggesters[language] = suggester
    suggester._prefetch([])  # the first word typed always has an empty context
    logger.info("[%s] Model ready (%d vocab words).", language, len(vocab))
    return suggester


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_completion(prompt: str, language: str = "tanglish", max_new_tokens: int = 15, **kwargs) -> str:
    """Compatibility method for the old /complete route."""
    suggester = _load_model(language)
    suggestions = suggester.suggest(prompt, k=1)
    return suggestions[0] if suggestions else ""


def generate_suggestions(prompt: str, language: str = "tanglish", k: int = 3) -> list[str]:
    """Return top k next-word / word-completion suggestions using the hybrid N-gram + LM model."""
    suggester = _load_model(language)
    return suggester.suggest(prompt, k=k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    test_prompts = {
        "tanglish": ["naan gym ", "naan gy", "enna ", "machi "],
        "tenglish": ["nuvvu ", "em ", "repu ", "nenu "],
    }

    for lang, prompts in test_prompts.items():
        print(f"\n── {lang} completion test ──────────────────────────")
        for p in prompts:
            c = generate_suggestions(p, language=lang)
            print(f"  [{p!r}]  →  {c!r}")
