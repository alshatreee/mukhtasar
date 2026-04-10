#!/usr/bin/env python3
"""Example: Using mukhtasar's Arabic NLP tools for kazima-ai.

Shows how to use the arabic module directly for:
- Text normalization before embedding/RAG
- Tokenization for search
- Summarizing long historical texts before indexing
"""

from mukhtasar.arabic import normalize, tokenize, split_sentences, light_stem, STOPWORDS
from mukhtasar import summarize

# ── 1. Normalize text before embedding ──────────────────────────
# Different spellings of the same word should match in search/RAG.

variants = ["إبراهيم", "ابراهيم", "أبراهيم"]
normalized = [normalize(v) for v in variants]
print("=== Normalization ===")
print(f"Input:  {variants}")
print(f"Output: {normalized}")
print(f"All equal: {len(set(normalized)) == 1}")  # True!

# ── 2. Tokenize for better search ──────────────────────────────
# Remove stopwords + apply stemming = better search results.

text = "المملكة العربية السعودية تقود التحول الرقمي في منطقة الشرق الأوسط"
raw_words = text.split()
clean_tokens = tokenize(text)

print(f"\n=== Tokenization ===")
print(f"Raw words ({len(raw_words)}):   {raw_words}")
print(f"Clean tokens ({len(clean_tokens)}): {clean_tokens}")

# ── 3. Summarize long historical text for RAG chunks ───────────
# Instead of putting 5000 words into a RAG chunk, summarize first.

historical_text = """
أحمد الجابر الصباح هو الحاكم الثالث عشر للكويت وقد حكم البلاد منذ عام 1921 حتى وفاته عام 1950.
يعتبر المؤسس الفعلي للكويت الحديثة حيث قاد البلاد خلال فترة التحول من الاقتصاد التقليدي إلى اقتصاد النفط.
في عهده تم اكتشاف النفط في الكويت عام 1938 وبدأ التصدير الفعلي عام 1946.
أسس أول مدرسة نظامية في الكويت وهي المدرسة المباركية عام 1911 قبل توليه الحكم.
كان يؤمن بأهمية التعليم والتحديث وأرسل البعثات الطلابية إلى الخارج.
أنشأ أول مستشفى حكومي وأول محطة إذاعة في الكويت.
عقد اتفاقيات دولية مهمة لحماية استقلال الكويت وسيادتها.
الخلاصة أن أحمد الجابر وضع الأساس لنهضة الكويت الحديثة في جميع المجالات.
"""

summary = summarize(historical_text, ratio=0.3, title="أحمد الجابر الصباح")

print(f"\n=== RAG Chunk Optimization ===")
print(f"Original: {len(historical_text)} chars")
print(f"Summary:  {len(summary.text)} chars ({summary.ratio:.0%})")
print(f"\nSummary for RAG:")
print(summary.text)

# ── 4. Process a batch of personality entries ───────────────────

personalities = [
    {"name": "أحمد الجابر", "bio": historical_text},
    {"name": "عبدالله السالم", "bio": "عبدالله السالم الصباح حاكم الكويت الذي قاد البلاد نحو الاستقلال. في عهده صدر الدستور الكويتي عام 1962."},
]

print(f"\n=== Batch Processing ===")
for p in personalities:
    tokens = tokenize(p["bio"])
    summary = summarize(p["bio"], ratio=0.4)
    print(f"\n{p['name']}:")
    print(f"  Keywords: {tokens[:5]}")
    print(f"  Summary: {summary.text[:80]}...")
