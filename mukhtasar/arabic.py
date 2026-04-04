"""Arabic text processing utilities — stopwords, sentence splitting, normalization."""

from __future__ import annotations

import re
import unicodedata

# ── Arabic stopwords (220+) ──────────────────────────────────────

STOPWORDS: set[str] = {
    # Pronouns
    "أنا", "أنت", "أنتِ", "أنتم", "أنتن", "نحن", "هو", "هي", "هم", "هن",
    # Prepositions
    "في", "من", "إلى", "على", "عن", "مع", "بين", "حتى", "منذ", "خلال",
    "عند", "لدى", "نحو", "فوق", "تحت", "أمام", "خلف", "حول", "دون", "بدون",
    # Conjunctions
    "و", "أو", "ثم", "ف", "لكن", "بل", "حيث", "إذ", "إذا", "لو", "كي",
    "لأن", "بسبب", "رغم", "مع أن", "بينما", "كما", "أيضا", "أيضاً",
    # Demonstratives
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء", "أولئك",
    # Relatives
    "الذي", "التي", "الذين", "اللذان", "اللتان", "اللواتي",
    # Question words
    "ما", "ماذا", "من", "أين", "متى", "كيف", "لماذا", "كم", "أي", "هل",
    # Verbs (common auxiliaries)
    "كان", "كانت", "كانوا", "يكون", "تكون", "ليس", "ليست", "ليسوا",
    "هناك", "يوجد", "لا يوجد",
    # Particles
    "لا", "لم", "لن", "قد", "سوف", "سـ", "إن", "أن", "كل", "بعض",
    "غير", "كثير", "قليل", "أكثر", "أقل", "جدا", "جداً", "فقط",
    # Common words
    "عام", "عامة", "خاص", "خاصة", "أول", "آخر", "جديد", "جديدة",
    "كبير", "كبيرة", "صغير", "صغيرة", "طويل", "قصير",
    "يجب", "يمكن", "ينبغي", "يستطيع",
    "بعد", "قبل", "أثناء", "حين", "عندما",
    "ذات", "نفس", "مثل", "عبر",
    # Numbers
    "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة",
    # Dialectal common (Gulf/Egyptian/Levantine)
    "هاي", "هاذا", "هاذي", "ذا", "كذا", "يعني", "بس", "لسه", "هسه",
    "شو", "ايش", "وش", "ليش", "كيذا", "زي", "عشان", "علشان",
}

# ── Normalization ─────────────────────────────────────────────────

_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
_TATWEEL = re.compile(r"\u0640+")
_ALEF_VARIANTS = re.compile(r"[إأآا]")
_TEH_MARBUTA = re.compile(r"ة")
_YEH_VARIANTS = re.compile(r"[ىي]")


def normalize(text: str) -> str:
    """Normalize Arabic text for comparison: strip diacritics, unify letters."""
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _TEH_MARBUTA.sub("ه", text)
    text = _YEH_VARIANTS.sub("ي", text)
    return text


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text))


# ── Sentence splitting ────────────────────────────────────────────

# Arabic sentence endings: period, Arabic question mark, exclamation,
# Arabic semicolon, newlines with content
_SENT_SPLIT = re.compile(r"(?<=[.!?؟،؛\n])\s+")


def split_sentences(text: str) -> list[str]:
    """Split Arabic text into sentences."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Split on sentence boundaries
    raw = _SENT_SPLIT.split(text)

    sentences = []
    for s in raw:
        s = s.strip()
        if len(s) > 10:  # Skip fragments
            sentences.append(s)

    # If no splits found, try splitting on newlines
    if len(sentences) <= 1 and "\n" in text:
        sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 10]

    return sentences


def tokenize(text: str) -> list[str]:
    """Tokenize Arabic text into words, removing stopwords and punctuation."""
    normalized = normalize(text)
    words = re.findall(r"[\u0600-\u06FF\u0750-\u077F]+", normalized)
    return [w for w in words if w not in STOPWORDS and len(w) > 1]
