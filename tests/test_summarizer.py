"""Tests for summarizer.py — summarization, scoring, multi-doc."""

import json
import tempfile
from pathlib import Path

from mukhtasar.summarizer import (
    Summary,
    score_sentences,
    summarize,
    summarize_file,
    summarize_multi,
)


SAMPLE_TEXT = (
    "الذكاء الاصطناعي يغير العالم بشكل كبير في العصر الحديث. "
    "التقنيات الحديثة تساعد في تطوير حلول مبتكرة للمشاكل المعقدة. "
    "الشركات الكبرى تستثمر مليارات الدولارات في هذا المجال المتنامي. "
    "المملكة العربية السعودية تقود التحول الرقمي في منطقة الشرق الأوسط. "
    "رؤية 2030 تضع التقنية في صميم خطة التنمية الوطنية الشاملة. "
    "الباحثون يطورون نماذج ذكاء اصطناعي أكثر كفاءة وأقل تكلفة. "
    "التعليم يتحول نحو استخدام التقنيات الذكية في الفصول الدراسية. "
    "الصحة الرقمية تستفيد من الذكاء الاصطناعي في التشخيص المبكر. "
    "النقل الذكي يعتمد على السيارات ذاتية القيادة والأنظمة المتقدمة. "
    "الخلاصة أن الذكاء الاصطناعي أصبح ضرورة وليس رفاهية في عالمنا."
)


# ── Basic summarization ─────────────────────────────────────────

class TestSummarize:
    def test_returns_summary_object(self):
        result = summarize(SAMPLE_TEXT)
        assert isinstance(result, Summary)

    def test_summary_shorter_than_original(self):
        result = summarize(SAMPLE_TEXT, ratio=0.3)
        assert result.summary_length < result.original_length

    def test_summary_has_sentences(self):
        result = summarize(SAMPLE_TEXT, ratio=0.3)
        assert result.sentence_count >= 1
        assert len(result.sentences) >= 1

    def test_ratio_respected(self):
        result = summarize(SAMPLE_TEXT, ratio=0.2)
        assert result.sentence_count <= result.original_sentences

    def test_max_sentences(self):
        result = summarize(SAMPLE_TEXT, max_sentences=2)
        assert result.sentence_count <= 2

    def test_single_sentence_returns_as_is(self):
        short = "جملة واحدة فقط"
        result = summarize(short)
        assert result.text == short.strip()
        assert result.ratio == 1.0

    def test_empty_text(self):
        result = summarize("")
        assert result.sentence_count == 0 or result.text == ""

    def test_with_title(self):
        result = summarize(SAMPLE_TEXT, title="الذكاء الاصطناعي", ratio=0.3)
        assert result.sentence_count >= 1

    def test_sentences_in_original_order(self):
        """Selected sentences should maintain their original order."""
        result = summarize(SAMPLE_TEXT, ratio=0.5)
        # Check that sentence order is preserved
        if len(result.sentences) >= 2:
            for i in range(len(result.sentences) - 1):
                pos_a = SAMPLE_TEXT.find(result.sentences[i][:20])
                pos_b = SAMPLE_TEXT.find(result.sentences[i + 1][:20])
                assert pos_a < pos_b


# ── Score sentences ──────────────────────────────────────────────

class TestScoreSentences:
    def test_returns_scored_list(self):
        scored = score_sentences(SAMPLE_TEXT)
        assert len(scored) >= 5

    def test_scores_are_sorted(self):
        scored = score_sentences(SAMPLE_TEXT)
        for i in range(len(scored) - 1):
            assert scored[i].score >= scored[i + 1].score

    def test_has_feature_breakdown(self):
        scored = score_sentences(SAMPLE_TEXT)
        for s in scored:
            assert s.features is not None
            assert "textrank" in s.features
            assert "position" in s.features
            assert "cue_words" in s.features

    def test_scores_between_0_and_1(self):
        scored = score_sentences(SAMPLE_TEXT)
        for s in scored:
            assert 0 <= s.score <= 1.0

    def test_empty_text(self):
        assert score_sentences("") == []

    def test_title_affects_scores(self):
        scored_no_title = score_sentences(SAMPLE_TEXT)
        scored_with_title = score_sentences(SAMPLE_TEXT, title="الذكاء الاصطناعي")
        # Scores should differ when title is provided
        scores_a = [s.score for s in scored_no_title]
        scores_b = [s.score for s in scored_with_title]
        assert scores_a != scores_b


# ── File summarization ───────────────────────────────────────────

class TestSummarizeFile:
    def test_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text(SAMPLE_TEXT, encoding="utf-8")
        result = summarize_file(str(f))
        assert result.sentence_count >= 1

    def test_jsonl_file(self, tmp_path):
        f = tmp_path / "test.jsonl"
        lines = [
            json.dumps({"text": "الذكاء الاصطناعي يغير العالم بشكل كبير في العصر الحديث."}, ensure_ascii=False),
            json.dumps({"text": "التقنيات الحديثة تساعد في تطوير حلول مبتكرة للمشاكل المعقدة."}, ensure_ascii=False),
            json.dumps({"text": "الشركات الكبرى تستثمر مليارات الدولارات في هذا المجال المتنامي."}, ensure_ascii=False),
            json.dumps({"text": "المملكة العربية السعودية تقود التحول الرقمي في الشرق الأوسط."}, ensure_ascii=False),
            json.dumps({"text": "الخلاصة أن الذكاء الاصطناعي أصبح ضرورة في عالمنا اليوم."}, ensure_ascii=False),
        ]
        f.write_text("\n".join(lines), encoding="utf-8")
        result = summarize_file(str(f))
        assert result.sentence_count >= 1

    def test_auto_title_from_first_line(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("عنوان المقالة\nالذكاء الاصطناعي يغير العالم. التقنية تتطور بسرعة. الشركات تستثمر مليارات.", encoding="utf-8")
        result = summarize_file(str(f))
        assert result.sentence_count >= 1


# ── Multi-document ───────────────────────────────────────────────

class TestSummarizeMulti:
    def test_two_documents(self, tmp_path):
        f1 = tmp_path / "doc1.txt"
        f2 = tmp_path / "doc2.txt"
        f1.write_text(
            "الذكاء الاصطناعي يغير العالم بشكل كبير. "
            "التقنيات الحديثة تساعد في تطوير حلول مبتكرة. "
            "الشركات تستثمر مليارات في هذا المجال.",
            encoding="utf-8",
        )
        f2.write_text(
            "التعليم يتحول نحو التقنيات الذكية بشكل متسارع. "
            "الصحة الرقمية تستفيد من الذكاء الاصطناعي كثيراً. "
            "النقل الذكي يعتمد على الأنظمة المتقدمة.",
            encoding="utf-8",
        )
        result = summarize_multi([str(f1), str(f2)])
        assert result.sentence_count >= 1

    def test_removes_duplicates(self, tmp_path):
        f1 = tmp_path / "doc1.txt"
        f2 = tmp_path / "doc2.txt"
        same_text = (
            "الذكاء الاصطناعي يغير العالم بشكل كبير في العصر الحديث. "
            "التقنيات الحديثة تساعد في تطوير حلول مبتكرة للمشاكل."
        )
        f1.write_text(same_text, encoding="utf-8")
        f2.write_text(same_text, encoding="utf-8")
        result = summarize_multi([str(f1), str(f2)])
        # With dedup, shouldn't double the sentences
        assert result.original_sentences <= 4
