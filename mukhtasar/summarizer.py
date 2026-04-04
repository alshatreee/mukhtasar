"""Extractive Arabic text summarizer using TextRank + TF-IDF."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from mukhtasar.arabic import is_arabic, normalize, split_sentences, tokenize


# ── Data structures ───────────────────────────────────────────────

@dataclass
class Summary:
    """Result of summarization."""
    original_length: int
    summary_length: int
    ratio: float
    sentence_count: int
    original_sentences: int
    sentences: list[str]
    text: str


@dataclass
class ScoredSentence:
    """A sentence with its importance score."""
    index: int
    text: str
    score: float


# ── TF-IDF computation ───────────────────────────────────────────

def _compute_tf(words: list[str]) -> dict[str, float]:
    """Term frequency for a single document (sentence)."""
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    total = len(words) if words else 1
    return {w: c / total for w, c in counts.items()}


def _compute_idf(tokenized_sentences: list[list[str]]) -> dict[str, float]:
    """Inverse document frequency across all sentences."""
    n = len(tokenized_sentences)
    if n == 0:
        return {}
    doc_freq: dict[str, int] = {}
    for tokens in tokenized_sentences:
        seen = set(tokens)
        for w in seen:
            doc_freq[w] = doc_freq.get(w, 0) + 1
    return {w: math.log(n / (1 + df)) for w, df in doc_freq.items()}


# ── Similarity matrix ────────────────────────────────────────────

def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    common = set(vec_a.keys()) & set(vec_b.keys())
    if not common:
        return 0.0
    dot = sum(vec_a[w] * vec_b[w] for w in common)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _build_tfidf_vectors(
    tokenized_sentences: list[list[str]],
    idf: dict[str, float],
) -> list[dict[str, float]]:
    """Build TF-IDF vector for each sentence."""
    vectors = []
    for tokens in tokenized_sentences:
        tf = _compute_tf(tokens)
        tfidf = {w: tf_val * idf.get(w, 0) for w, tf_val in tf.items()}
        vectors.append(tfidf)
    return vectors


# ── TextRank ──────────────────────────────────────────────────────

def _textrank(
    similarity_matrix: list[list[float]],
    damping: float = 0.85,
    iterations: int = 30,
    convergence: float = 0.0001,
) -> list[float]:
    """TextRank algorithm on a similarity matrix. Returns scores per node."""
    n = len(similarity_matrix)
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
                # Weighted edge: similarity(i,j)
                out_sum = sum(similarity_matrix[j][k] for k in range(n) if k != j)
                if out_sum > 0:
                    rank_sum += similarity_matrix[j][i] / out_sum * scores[j]
            new_scores[i] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        delta = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if delta < convergence:
            break

    return scores


# ── Public API ────────────────────────────────────────────────────

def summarize(
    text: str,
    ratio: float = 0.3,
    max_sentences: int | None = None,
    min_sentences: int = 1,
) -> Summary:
    """
    Summarize Arabic text using extractive TextRank + TF-IDF.

    Args:
        text: Input text (Arabic or mixed)
        ratio: Target summary length as fraction of original (0.0-1.0)
        max_sentences: Hard cap on output sentences
        min_sentences: Minimum sentences to return

    Returns:
        Summary with ranked sentences in original order
    """
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

    # Tokenize each sentence
    tokenized = [tokenize(s) for s in sentences]

    # Build TF-IDF
    idf = _compute_idf(tokenized)
    vectors = _build_tfidf_vectors(tokenized, idf)

    # Build similarity matrix
    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # Run TextRank
    scores = _textrank(sim_matrix)

    # Score sentences
    scored = [ScoredSentence(i, sentences[i], scores[i]) for i in range(n)]

    # Determine how many to keep
    target = max(min_sentences, int(len(sentences) * ratio))
    if max_sentences:
        target = min(target, max_sentences)
    target = min(target, len(sentences))

    # Select top-scored sentences
    ranked = sorted(scored, key=lambda s: s.score, reverse=True)[:target]

    # Return in original order
    selected = sorted(ranked, key=lambda s: s.index)
    summary_sentences = [s.text for s in selected]
    summary_text = " ".join(summary_sentences)

    return Summary(
        original_length=len(text),
        summary_length=len(summary_text),
        ratio=len(summary_text) / len(text) if text else 0,
        sentence_count=len(summary_sentences),
        original_sentences=len(sentences),
        sentences=summary_sentences,
        text=summary_text,
    )


def summarize_file(
    path: str | Path,
    ratio: float = 0.3,
    max_sentences: int | None = None,
) -> Summary:
    """Summarize text from a file."""
    text = Path(path).read_text(encoding="utf-8")

    # Handle JSONL: summarize the 'text' field of each line
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

    return summarize(text, ratio=ratio, max_sentences=max_sentences)


def score_sentences(text: str) -> list[ScoredSentence]:
    """Return all sentences with their TextRank scores, sorted by score."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    tokenized = [tokenize(s) for s in sentences]
    idf = _compute_idf(tokenized)
    vectors = _build_tfidf_vectors(tokenized, idf)

    n = len(sentences)
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(vectors[i], vectors[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    scores = _textrank(sim_matrix)
    scored = [ScoredSentence(i, sentences[i], scores[i]) for i in range(n)]
    return sorted(scored, key=lambda s: s.score, reverse=True)
