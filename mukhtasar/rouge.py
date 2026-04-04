"""ROUGE evaluation for Arabic summaries — lemma-aware for morphological richness."""

from __future__ import annotations

from dataclasses import dataclass

from mukhtasar.arabic import light_stem, normalize


@dataclass
class RougeScores:
    rouge1: RougeMetric
    rouge2: RougeMetric
    rougeL: RougeMetric


@dataclass
class RougeMetric:
    precision: float
    recall: float
    f1: float


def _tokenize_for_rouge(text: str, use_stems: bool = True) -> list[str]:
    """Tokenize text for ROUGE comparison. Uses light stemming for Arabic."""
    normalized = normalize(text)
    # Split on whitespace and punctuation
    import re
    words = re.findall(r"[\w\u0600-\u06FF\u0750-\u077F]+", normalized.lower())
    if use_stems:
        return [light_stem(w) for w in words if len(w) > 1]
    return [w for w in words if len(w) > 1]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Longest Common Subsequence length."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Space-optimized LCS
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _compute_rouge_n(reference_tokens: list[str], summary_tokens: list[str], n: int) -> RougeMetric:
    """Compute ROUGE-N (unigram or bigram overlap)."""
    ref_ngrams = _ngrams(reference_tokens, n)
    sum_ngrams = _ngrams(summary_tokens, n)

    if not ref_ngrams or not sum_ngrams:
        return RougeMetric(0.0, 0.0, 0.0)

    ref_counts: dict[tuple[str, ...], int] = {}
    for ng in ref_ngrams:
        ref_counts[ng] = ref_counts.get(ng, 0) + 1

    sum_counts: dict[tuple[str, ...], int] = {}
    for ng in sum_ngrams:
        sum_counts[ng] = sum_counts.get(ng, 0) + 1

    # Count overlapping n-grams
    overlap = 0
    for ng, count in sum_counts.items():
        if ng in ref_counts:
            overlap += min(count, ref_counts[ng])

    precision = overlap / len(sum_ngrams) if sum_ngrams else 0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return RougeMetric(round(precision, 4), round(recall, 4), round(f1, 4))


def _compute_rouge_l(reference_tokens: list[str], summary_tokens: list[str]) -> RougeMetric:
    """Compute ROUGE-L (Longest Common Subsequence)."""
    if not reference_tokens or not summary_tokens:
        return RougeMetric(0.0, 0.0, 0.0)

    lcs = _lcs_length(reference_tokens, summary_tokens)
    precision = lcs / len(summary_tokens) if summary_tokens else 0
    recall = lcs / len(reference_tokens) if reference_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return RougeMetric(round(precision, 4), round(recall, 4), round(f1, 4))


def evaluate(reference: str, summary: str, use_stems: bool = True) -> RougeScores:
    """Evaluate summary against reference using ROUGE-1, ROUGE-2, ROUGE-L.

    Uses Arabic light stemming by default (LEMMA-ROUGE approach) for better
    handling of Arabic morphological complexity.
    """
    ref_tokens = _tokenize_for_rouge(reference, use_stems=use_stems)
    sum_tokens = _tokenize_for_rouge(summary, use_stems=use_stems)

    return RougeScores(
        rouge1=_compute_rouge_n(ref_tokens, sum_tokens, 1),
        rouge2=_compute_rouge_n(ref_tokens, sum_tokens, 2),
        rougeL=_compute_rouge_l(ref_tokens, sum_tokens),
    )
