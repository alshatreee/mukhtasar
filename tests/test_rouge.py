"""Tests for rouge.py — ROUGE evaluation for Arabic text."""

from mukhtasar.rouge import RougeMetric, RougeScores, evaluate


REFERENCE = (
    "الذكاء الاصطناعي يغير العالم بشكل كبير. "
    "الشركات تستثمر مليارات في تطوير التقنيات الحديثة. "
    "المملكة العربية السعودية تقود التحول الرقمي."
)

# Same text = perfect score
PERFECT_SUMMARY = REFERENCE

# Partial overlap
PARTIAL_SUMMARY = (
    "الذكاء الاصطناعي يغير العالم بشكل كبير. "
    "التعليم يتطور بسرعة كبيرة في المنطقة."
)

# No overlap
UNRELATED_SUMMARY = "الطقس جميل اليوم والسماء صافية تماماً"


class TestRougeEvaluate:
    def test_returns_rouge_scores(self):
        scores = evaluate(REFERENCE, PARTIAL_SUMMARY)
        assert isinstance(scores, RougeScores)
        assert isinstance(scores.rouge1, RougeMetric)
        assert isinstance(scores.rouge2, RougeMetric)
        assert isinstance(scores.rougeL, RougeMetric)

    def test_perfect_match_high_score(self):
        scores = evaluate(REFERENCE, PERFECT_SUMMARY)
        assert scores.rouge1.f1 > 0.9
        assert scores.rougeL.f1 > 0.9

    def test_partial_overlap_medium_score(self):
        scores = evaluate(REFERENCE, PARTIAL_SUMMARY)
        assert 0.1 < scores.rouge1.f1 < 0.9

    def test_no_overlap_low_score(self):
        scores = evaluate(REFERENCE, UNRELATED_SUMMARY)
        assert scores.rouge1.f1 < 0.3

    def test_scores_between_0_and_1(self):
        scores = evaluate(REFERENCE, PARTIAL_SUMMARY)
        for metric in [scores.rouge1, scores.rouge2, scores.rougeL]:
            assert 0 <= metric.precision <= 1
            assert 0 <= metric.recall <= 1
            assert 0 <= metric.f1 <= 1

    def test_with_stemming(self):
        # Use text with clear morphological variation
        ref = "المطورون يكتبون البرامج المتقدمة والتطبيقات الحديثة"
        summary = "المطورات كتبن البرامج المتقدمات والتطبيق الحديث"
        scores_stem = evaluate(ref, summary, use_stems=True)
        scores_no_stem = evaluate(ref, summary, use_stems=False)
        # With stemming, inflected forms match better → higher score
        assert scores_stem.rouge1.f1 >= scores_no_stem.rouge1.f1

    def test_empty_reference(self):
        scores = evaluate("", PARTIAL_SUMMARY)
        assert scores.rouge1.f1 == 0.0

    def test_empty_summary(self):
        scores = evaluate(REFERENCE, "")
        assert scores.rouge1.f1 == 0.0
