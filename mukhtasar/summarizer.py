"""Extractive Arabic summarizer — TextRank + TF-IDF + multi-feature scoring."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mukhtasar.arabic import (
    CUE_WORDS_IMPORTANT,
    CUE_WORDS_UNIMPORTANT,
    count_proper_nouns,
    has_number,
    light_stem,
    normalize,
    split_sentences,
    tokenize,
    tokenize_raw,
)


# ── Data structures ───────────────────────────────────────────────

@dataclass
class Summary:
    original_length: int
    summary_length: int
    ratio: float
    sentence_count: int
    original_sentences: int
    sentences: list[str]
    text: str


@dataclass
class ScoredSentence:
    index: int
    text: str
    score: float
    features: dict[str, float] | None = None


# ── TF-IDF ────────────────────────────────────────────────────────

def _compute_tf(words: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    total = len(words) if words else 1
    return {w: c / total for w, c in counts.items()}


def _compute_idf(tokenized_sentences: list[list[str]]) -> dict[str, float]:
    n = len(tokenized_sentences)
    if n == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for tokens in tokenized_sentences:
        for w in set(tokens):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    return {w: math.log(n / (1 + df)) for w, df in doc_freq.items()}


# ── Similarity + TextRank ─────────────────────────────────────────

def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a.keys()) & set(b.keys())
    if not common:
        return 0.0
    dot = sum(a[w] * b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _textrank(sim_matrix: list[list[float]], damping: float = 0.85, iterations: int = 30) -> list[float]:
    n = len(sim_matrix)
    if n == 0:
        return []
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if i == j:
                    continue
                out_sum = sum(sim_matrix[j][k] for k in range(n) if k != j)
                if out_sum > 0:
                    rank_sum += sim_matrix[j][i] / out_sum * scores[j]
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if delta < 0.0001:
            break
    return scores


# ── Feature scoring ───────────────────────────────────────────────

def _position_score(index: int, total: int) -> float:
    """First and last sentences score higher. First 20% and last 10% boosted."""
    if total <= 1:
        return 1.0
    pos = index / total
    if pos < 0.2:
        return 1.0 - (pos * 2)  # 1.0 → 0.6
    if pos > 0.9:
        return 0.6 + (pos - 0.9) * 4  # 0.6 → 1.0
    return 0.3


def _length_score(sentence: str, avg_length: float) -> float:
    """Penalize very short and very long sentences. Optimal ~80-200 chars."""
    length = len(sentence)
    if length < 20:
        return 0.1
    if length < 40:
        return 0.3
    if avg_length > 0:
        ratio = length / avg_length
        if 0.5 <= ratio <= 2.0:
            return 1.0
        if ratio > 2.0:
            return 0.5
    return 0.6


def _cue_word_score(sentence: str) -> float:
    """Score based on Arabic cue words that signal importance."""
    raw_words = set(tokenize_raw(sentence))
    normalized_sentence = normalize(sentence)

    # Check important cue words
    important_hits = 0
    for cue in CUE_WORDS_IMPORTANT:
        if cue in normalized_sentence or cue in raw_words:
            important_hits += 1

    # Check unimportant cue words
    for cue in CUE_WORDS_UNIMPORTANT:
        if cue in normalized_sentence:
            important_hits -= 1

    if important_hits >= 3:
        return 1.0
    if important_hits >= 2:
        return 0.8
    if important_hits >= 1:
        return 0.6
    return 0.2


def _number_score(sentence: str) -> float:
    """Sentences with numbers/statistics tend to be important."""
    return 0.7 if has_number(sentence) else 0.3


def _proper_noun_score(sentence: str) -> float:
    """Sentences with more proper nouns tend to carry more information."""
    count = count_proper_nouns(sentence)
    if count >= 3:
        return 1.0
    if count >= 2:
        return 0.7
    if count >= 1:
        return 0.5
    return 0.2


def _title_similarity_score(sentence: str, title: str | None) -> float:
    """Score based on word overlap with the title/header."""
    if not title:
        return 0.5  # Neutral if no title

    title_words = set(tokenize(title))
    sent_words = set(tokenize(sentence))

    if not title_words:
        return 0.5

    overlap = len(title_words & sent_words)
    ratio = overlap / len(title_words)

    if ratio >= 0.5:
        return 1.0
    if ratio >= 0.25:
        return 0.7
    if overlap >= 1:
        return 0.5
    return 0.2


# ── Combined scoring ──────────────────────────────────────────────

# Feature weights (tuned for Arabic extractive summarization)
WEIGHTS = {
    "textrank": 0.35,
    "position": 0.15,
    "length": 0.05,
    "cue_words": 0.15,
    "numbers": 0.05,
    "proper_nouns": 0.10,
    "title_sim": 0.15,
}


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _combined_score(
    sentences: list[str],
    textrank_scores: list[float],
    title: str | None = None,
) -> list[ScoredSentence]:
    """Combine all features into a final score per sentence."""
    n = len(sentences)
    avg_length = sum(len(s) for s in sentences) / n if n else 100

    # Normalize TextRank scores
    tr_norm = _normalize_scores(textrank_scores)

    scored = []
    for i in range(n):
        features = {
            "textrank": tr_norm[i],
            "position": _position_score(i, n),
            "length": _length_score(sentences[i], avg_length),
            "cue_words": _cue_word_score(sentences[i]),
            "numbers": _number_score(sentences[i]),
            "proper_nouns": _proper_noun_score(sentences[i]),
            "title_sim": _title_similarity_score(sentences[i], title),
        }

        final = sum(WEIGHTS[k] * features[k] for k in WEIGHTS)
        scored.append(ScoredSentence(i, sentences[i], final, features))

    return scored


# ── Public API ────────────────────────────────────────────────────

def summarize(
    text: str,
    ratio: float = 0.3,
    max_sentences: int | None = None,
    min_sentences: int = 1,
    title: str | None = None,
) -> Summary:
    """Summarize Arabic text using TextRank + TF-IDF + multi-feature scoring."""
    sentences = split_sentences(text)

    if len(sentences) <= min_sentences:
        return Summary(
            original_length=len(text),
            summary_length=len(text),
            ratio=1.0,
            sentence_count=len(sentences),
            original_sentences=len(sentences),
            sentences=sentences,
            text=text.strip(),
        )

    # Tokenize with stemming
    tokenized = [tokenize(s) for s in sentences]

    # TF-IDF
    idf = _compute_idf(tokenized)
    vectors = []
    for tokens in tokenized:
        tf = _compute_tf(tokens)
        vectors.append({w: tf[w] * idf.get(w, 0) for w in tf})

    # Similarity matrix
    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # TextRank
    tr_scores = _textrank(sim_matrix)

    # Combined multi-feature scoring
    scored = _combined_score(sentences, tr_scores, title=title)

    # Select top sentences
    target = max(min_sentences, int(n * ratio))
    if max_sentences:
        target = min(target, max_sentences)
    target = min(target, n)

    ranked = sorted(scored, key=lambda s: s.score, reverse=True)[:target]
    selected = sorted(ranked, key=lambda s: s.index)
    summary_sentences = [s.text for s in selected]
    summary_text = " ".join(summary_sentences)

    return Summary(
        original_length=len(text),
        summary_length=len(summary_text),
        ratio=len(summary_text) / len(text) if text else 0,
        sentence_count=len(summary_sentences),
        original_sentences=n,
        sentences=summary_sentences,
        text=summary_text,
    )


def summarize_file(
    path: str | Path,
    ratio: float = 0.3,
    max_sentences: int | None = None,
    title: str | None = None,
) -> Summary:
    """Summarize text from a file (.txt or .jsonl)."""
    text = Path(path).read_text(encoding="utf-8")

    if str(path).endswith(".jsonl"):
        lines = []
        for line in text.strip().split("\n"):
            try:
                obj = json.loads(line)
                content = obj.get("text") or obj.get("content") or obj.get("body") or ""
                if content:
                    lines.append(content)
            except json.JSONDecodeError:
                continue
        text = "\n".join(lines)

    # Try to extract title from first line
    if not title:
        first_line = text.strip().split("\n")[0].strip()
        if len(first_line) < 100 and not first_line.endswith("."):
            title = first_line

    return summarize(text, ratio=ratio, max_sentences=max_sentences, title=title)


def summarize_multi(
    paths: list[str | Path],
    ratio: float = 0.3,
    max_sentences: int | None = None,
) -> Summary:
    """Summarize across multiple documents with redundancy removal."""
    all_sentences: list[str] = []
    all_text = ""

    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        all_text += text + "\n"
        all_sentences.extend(split_sentences(text))

    if not all_sentences:
        return Summary(0, 0, 0, 0, 0, [], "")

    # Remove near-duplicate sentences (>80% word overlap)
    unique: list[str] = []
    unique_tokens: list[set[str]] = []
    for s in all_sentences:
        tokens = set(tokenize(s))
        is_dup = False
        for existing in unique_tokens:
            if existing and tokens:
                overlap = len(tokens & existing) / max(len(tokens), len(existing))
                if overlap > 0.8:
                    is_dup = True
                    break
        if not is_dup:
            unique.append(s)
            unique_tokens.append(tokens)

    # Summarize the deduplicated set
    combined = "\n".join(unique)
    return summarize(combined, ratio=ratio, max_sentences=max_sentences)


def score_sentences(text: str, title: str | None = None) -> list[ScoredSentence]:
    """Return all sentences scored and ranked, with feature breakdown."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    tokenized = [tokenize(s) for s in sentences]
    idf = _compute_idf(tokenized)
    vectors = []
    for tokens in tokenized:
        tf = _compute_tf(tokens)
        vectors.append({w: tf[w] * idf.get(w, 0) for w in tf})

    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    tr_scores = _textrank(sim_matrix)
    scored = _combined_score(sentences, tr_scores, title=title)
    return sorted(scored, key=lambda s: s.score, reverse=True)
