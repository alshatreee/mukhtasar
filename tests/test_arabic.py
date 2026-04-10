"""Tests for arabic.py — normalization, tokenization, stemming, sentence splitting."""

from mukhtasar.arabic import (
    CUE_WORDS_IMPORTANT,
    STOPWORDS,
    count_proper_nouns,
    has_number,
    is_arabic,
    light_stem,
    normalize,
    split_sentences,
    tokenize,
    tokenize_raw,
)


# ── Normalization ────────────────────────────────────────────────

class TestNormalize:
    def test_removes_diacritics(self):
        assert normalize("كِتَابٌ") == "كتاب"

    def test_unifies_alef(self):
        assert normalize("إبراهيم") == normalize("ابراهيم")
        assert normalize("أحمد") == normalize("احمد")
        assert normalize("آمال") == normalize("امال")

    def test_unifies_teh_marbuta(self):
        assert normalize("مدرسة") == normalize("مدرسه")

    def test_unifies_yeh(self):
        assert normalize("على") == normalize("علي")

    def test_removes_tatweel(self):
        assert normalize("كتـــاب") == "كتاب"

    def test_empty_string(self):
        assert normalize("") == ""

    def test_english_unchanged(self):
        assert normalize("hello") == "hello"


# ── is_arabic ────────────────────────────────────────────────────

class TestIsArabic:
    def test_arabic_text(self):
        assert is_arabic("مرحبا بالعالم")

    def test_english_text(self):
        assert not is_arabic("hello world")

    def test_mixed_text(self):
        assert is_arabic("hello مرحبا world")

    def test_empty_string(self):
        assert not is_arabic("")


# ── Light Stemmer ────────────────────────────────────────────────

class TestLightStem:
    def test_removes_al_prefix(self):
        assert light_stem("الكتاب") == "كتاب"

    def test_removes_wal_prefix(self):
        assert light_stem("والكتاب") == "كتاب"

    def test_removes_plural_suffix(self):
        assert light_stem("كتابات") == "كتاب"
        assert light_stem("مطورات") == "مطور"

    def test_removes_oon_suffix(self):
        assert light_stem("مطورون") == "مطور"

    def test_short_word_unchanged(self):
        assert light_stem("من") == "من"
        assert light_stem("في") == "في"

    def test_preserves_minimum_length(self):
        # Should not strip to less than 2 chars
        result = light_stem("الب")
        assert len(result) >= 2

    def test_no_over_strip_short_prefix(self):
        # ب, ك, ل, س should NOT be stripped as single-char prefixes
        assert light_stem("كتاب") == "كتاب"
        assert light_stem("بلاد") == "بلاد"
        assert light_stem("سيارة") == "سيار"  # only suffix stripped

    def test_conjunction_prefix_works(self):
        # و and ف ARE safe single-char prefixes (conjunctions)
        assert light_stem("وكتاب") == "كتاب"
        assert light_stem("فكتاب") == "كتاب"

    def test_long_prefix_works(self):
        assert light_stem("والكتاب") == "كتاب"
        assert light_stem("بالمدرسة") == "مدرس"

    def test_combined_forms(self):
        # Different forms of same root should converge
        stem_1 = light_stem("المطورون")
        stem_2 = light_stem("مطورات")
        assert stem_1 == stem_2  # both → مطور


# ── Sentence Splitting ──────────────────────────────────────────

class TestSplitSentences:
    def test_period_split(self):
        text = "الجملة الأولى هنا. الجملة الثانية هنا. الجملة الثالثة هنا."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_question_mark_split(self):
        text = "ما هو الذكاء الاصطناعي؟ هو علم يدرس الآلات الذكية."
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_arabic_comma_fallback(self):
        # If no periods, should fall back to Arabic comma
        text = "الذكاء الاصطناعي مهم جداً، والتقنية تتطور بسرعة، والشركات تستثمر فيه"
        sentences = split_sentences(text)
        assert len(sentences) >= 2

    def test_short_fragments_filtered(self):
        text = "كلمة. الجملة الثانية هنا طويلة بما يكفي."
        sentences = split_sentences(text)
        # "كلمة" is too short (< 10 chars), should be filtered
        for s in sentences:
            assert len(s) > 10

    def test_empty_text(self):
        assert split_sentences("") == []

    def test_single_sentence(self):
        text = "هذه جملة واحدة طويلة بما يكفي لتمريرها"
        sentences = split_sentences(text)
        assert len(sentences) == 1


# ── Tokenize ─────────────────────────────────────────────────────

class TestTokenize:
    def test_removes_stopwords(self):
        text = "هذا هو الكتاب الذي في المكتبة"
        tokens = tokenize(text)
        # "هذا", "هو", "الذي", "في" are stopwords — should be removed
        for stopword in ["هذا", "هو", "في"]:
            assert stopword not in tokens

    def test_applies_stemming(self):
        tokens = tokenize("الكتابات والمطورون")
        # Should have stemmed forms
        assert len(tokens) >= 1

    def test_empty_text(self):
        assert tokenize("") == []

    def test_english_only_returns_empty(self):
        assert tokenize("hello world python") == []


class TestTokenizeRaw:
    def test_no_stemming(self):
        tokens = tokenize_raw("الكتابات والمطورون")
        # Raw should keep original forms
        assert any("كتابات" in t or "الكتابات" in t for t in tokens)

    def test_filters_short_words(self):
        tokens = tokenize_raw("و في من الكتاب")
        for t in tokens:
            assert len(t) > 1


# ── has_number ───────────────────────────────────────────────────

class TestHasNumber:
    def test_western_digits(self):
        assert has_number("عام 2024 كان مهماً")

    def test_arabic_digits(self):
        assert has_number("العدد ٤٥ مهم")

    def test_no_digits(self):
        assert not has_number("لا أرقام هنا")


# ── count_proper_nouns ───────────────────────────────────────────

class TestCountProperNouns:
    def test_with_proper_nouns(self):
        text = "المملكة العربية السعودية تقود التحول الرقمي"
        count = count_proper_nouns(text)
        assert count >= 1

    def test_no_proper_nouns(self):
        text = "يعني بس كذا"
        count = count_proper_nouns(text)
        assert count == 0


# ── Stopwords coverage ──────────────────────────────────────────

class TestStopwords:
    def test_has_gulf_dialect(self):
        gulf_words = {"يعني", "بس", "وش", "ليش"}
        assert gulf_words.issubset(STOPWORDS)

    def test_has_egyptian_dialect(self):
        egyptian_words = {"كده", "دا", "دي", "مش"}
        assert egyptian_words.issubset(STOPWORDS)

    def test_has_msa_basics(self):
        msa_words = {"في", "من", "إلى", "على", "هذا", "هذه"}
        assert msa_words.issubset(STOPWORDS)

    def test_cue_words_exist(self):
        assert len(CUE_WORDS_IMPORTANT) > 10
