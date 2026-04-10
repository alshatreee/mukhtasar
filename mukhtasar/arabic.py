"""Arabic text processing — stopwords, sentence splitting, normalization, root extraction."""

from __future__ import annotations

import re

# ── Arabic stopwords (250+) ──────────────────────────────────────

STOPWORDS: set[str] = {
    # Pronouns
    "أنا", "أنت", "أنتِ", "أنتم", "أنتن", "نحن", "هو", "هي", "هم", "هن",
    # Prepositions
    "في", "من", "إلى", "على", "عن", "مع", "بين", "حتى", "منذ", "خلال",
    "عند", "لدى", "نحو", "فوق", "تحت", "أمام", "خلف", "حول", "دون", "بدون",
    "ضد", "وسط", "عبر", "ضمن", "لدى",
    # Conjunctions
    "و", "أو", "ثم", "ف", "لكن", "بل", "حيث", "إذ", "إذا", "لو", "كي",
    "لأن", "بسبب", "رغم", "مع أن", "بينما", "كما", "أيضا", "أيضاً",
    # Demonstratives
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "أولئك",
    # Relatives
    "الذي", "التي", "الذين", "اللذان", "اللتان", "اللواتي",
    # Question words
    "ما", "ماذا", "أين", "متى", "كيف", "لماذا", "كم", "أي", "هل",
    # Verbs (auxiliaries)
    "كان", "كانت", "كانوا", "يكون", "تكون", "ليس", "ليست", "ليسوا",
    "هناك", "يوجد", "لا يوجد", "أصبح", "أصبحت", "صار", "بات",
    # Particles
    "لا", "لم", "لن", "قد", "سوف", "سـ", "إن", "أن", "كل", "بعض",
    "غير", "كثير", "قليل", "أكثر", "أقل", "جدا", "جداً", "فقط",
    "حيث", "عندما", "بعد", "قبل", "أثناء", "حين",
    "ذات", "نفس", "مثل", "مثلما",
    # Common adjectives
    "عام", "عامة", "خاص", "خاصة", "أول", "آخر", "جديد", "جديدة",
    "كبير", "كبيرة", "صغير", "صغيرة", "طويل", "قصير",
    "يجب", "يمكن", "ينبغي", "يستطيع",
    # Numbers
    "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة",
    # Dialectal (Gulf/Egyptian/Levantine)
    "هاي", "هاذا", "هاذي", "ذا", "كذا", "يعني", "بس", "لسه", "هسه",
    "شو", "ايش", "وش", "ليش", "كيذا", "زي", "عشان", "علشان",
    "كده", "دا", "دي", "دول", "مش", "حاجة", "برضو",
}

# ── Arabic cue words (signal important sentences) ─────────────────

CUE_WORDS_IMPORTANT: set[str] = {
    # Conclusion markers
    "الخلاصة", "خلاصة", "نستنتج", "نخلص", "في الختام", "ختاماً", "أخيراً",
    "في النهاية", "نهاية", "المحصلة", "باختصار", "إجمالاً",
    # Importance markers
    "أهم", "الأهم", "مهم", "أساسي", "رئيسي", "جوهري", "حاسم", "بارز",
    "ملحوظ", "لافت", "ضروري", "حيوي", "جذري",
    # Result markers
    "النتيجة", "نتيجة", "بالتالي", "لذلك", "وبالتالي", "ولذلك",
    "من ثم", "وعليه", "يترتب",
    # Summary markers
    "بشكل عام", "عموماً", "إجمالاً", "بصفة عامة", "بوجه عام",
    # Emphasis
    "يجدر", "تجدر", "يؤكد", "تؤكد", "أكد", "أثبت", "أظهر", "كشف",
    "أوضح", "بيّن", "أشار",
    # Comparison/contrast
    "على عكس", "بخلاف", "مقارنة", "بالمقارنة",
}

CUE_WORDS_UNIMPORTANT: set[str] = {
    "على سبيل المثال", "مثلاً", "كمثال", "منها", "من بينها",
}

# ── Normalization ─────────────────────────────────────────────────

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
_TATWEEL = re.compile(r"\u0640+")
_ALEF_VARIANTS = re.compile(r"[إأآا]")
_TEH_MARBUTA = re.compile(r"ة")
_YEH_VARIANTS = re.compile(r"[ىي]")


def normalize(text: str) -> str:
    """Normalize Arabic text for comparison."""
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _TEH_MARBUTA.sub("ه", text)
    text = _YEH_VARIANTS.sub("ي", text)
    return text


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))


# ── Root extraction (lightweight Arabic stemmer) ──────────────────

# Common Arabic prefixes and suffixes for light stemming
# Long prefixes (safe to remove with 2+ remaining chars)
_PREFIXES_LONG = ["وال", "بال", "كال", "فال", "لل", "ال"]
# Short prefixes — only conjunctions (و and ف) are safe as single-char prefixes.
# ب, ك, ل, س are too ambiguous alone (كتاب, بلاد, لعبة, سيارة) so they are
# only stripped as part of long prefixes (بال, كال, etc.)
_PREFIXES_SHORT = ["و", "ف"]
# Suffixes sorted longest-first
_SUFFIXES = ["ها", "هم", "هن", "كم", "كن", "نا", "ون", "ين", "ات", "ان", "تا", "تم", "تن", "ية", "ي", "ه", "ك", "ت", "ا", "ة", "ن"]


def light_stem(word: str) -> str:
    """Light Arabic stemmer — removes common prefixes and suffixes.

    Not a full root extractor, but maps inflected forms closer together:
    كتابات → كتاب, المطورون → مطور, يكتبون → يكتب

    Improved: single-char prefixes (و ف ب ك ل س) require at least
    3 remaining characters, preventing over-stripping like كتاب → تاب.
    """
    w = normalize(word)
    if len(w) <= 3:
        return w

    # Try long prefixes first (2+ chars, need 2+ remaining)
    stripped = False
    for prefix in _PREFIXES_LONG:
        if w.startswith(prefix) and len(w) - len(prefix) >= 2:
            w = w[len(prefix):]
            stripped = True
            break

    # Try short prefixes only if long prefix didn't match
    # Require 3+ remaining chars to avoid over-stripping
    if not stripped:
        for prefix in _PREFIXES_SHORT:
            if w.startswith(prefix) and len(w) - len(prefix) >= 3:
                w = w[len(prefix):]
                break

    # Remove suffixes (longest first, need 2+ remaining)
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 2:
            w = w[:-len(suffix)]
            break

    return w


# ── Sentence splitting ────────────────────────────────────────────

# Arabic sentence endings
_SENT_END = re.compile(r"(?<=[.!?؟])\s+")
_ARABIC_COMMA_SPLIT = re.compile(r"(?<=[،؛])\s+")
_BULLET_PATTERN = re.compile(r"^\s*[-•●▪◆★\d]+[.)]\s*", re.MULTILINE)
_QUOTED_SPEECH = re.compile(r'[«"](.*?)[»"]', re.DOTALL)


def split_sentences(text: str) -> list[str]:
    """Split Arabic text into sentences with improved handling."""
    text = re.sub(r"\s+", " ", text.strip())

    # Handle bullet points — each bullet is a sentence
    if _BULLET_PATTERN.search(text):
        lines = text.split("\n")
        sentences = []
        for line in lines:
            line = _BULLET_PATTERN.sub("", line).strip()
            if len(line) > 10:
                sentences.append(line)
        if sentences:
            return sentences

    # Split on sentence-ending punctuation
    raw = _SENT_END.split(text)

    # If too few splits, try Arabic comma/semicolon
    if len(raw) <= 2:
        expanded = []
        for chunk in raw:
            parts = _ARABIC_COMMA_SPLIT.split(chunk)
            expanded.extend(parts)
        raw = expanded

    # If still too few, try newlines
    if len(raw) <= 2 and "\n" in text:
        raw = text.split("\n")

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 10:
            sentences.append(s)

    return sentences


def tokenize(text: str) -> list[str]:
    """Tokenize Arabic text, remove stopwords, apply light stemming."""
    normalized = normalize(text)
    words = re.findall(r"[\u0600-\u06FF\u0750-\u077F]+", normalized)
    return [light_stem(w) for w in words if w not in STOPWORDS and len(w) > 1]


def tokenize_raw(text: str) -> list[str]:
    """Tokenize without stemming — for cue word and proper noun detection."""
    words = re.findall(r"[\u0600-\u06FF\u0750-\u077F]+", text)
    return [w for w in words if len(w) > 1]


def has_number(text: str) -> bool:
    """Check if text contains numbers (Arabic or Western)."""
    return bool(re.search(r"[\d٠-٩]", text))


def count_proper_nouns(text: str) -> int:
    """Estimate proper noun count — words with ال that aren't stopwords and are capitalized-equivalent."""
    words = re.findall(r"[\u0600-\u06FF\u0750-\u077F]+", text)
    count = 0
    for w in words:
        # Arabic proper nouns often start with ال and aren't common words
        if w.startswith("ال") and w not in STOPWORDS and len(w) > 3:
            stem = light_stem(w)
            if stem not in STOPWORDS:
                count += 1
    return count
