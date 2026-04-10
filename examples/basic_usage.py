#!/usr/bin/env python3
"""Basic usage examples for mukhtasar as a Python library.

Shows how to use mukhtasar in your own Python code (not just CLI).
Useful for integrating into kazima-ai, bots, or any Arabic NLP pipeline.
"""

from mukhtasar import summarize, score_sentences, evaluate

# ── 1. Summarize Arabic text ────────────────────────────────────

text = """
الذكاء الاصطناعي يغير العالم بشكل كبير في العصر الحديث.
التقنيات الحديثة تساعد في تطوير حلول مبتكرة للمشاكل المعقدة.
الشركات الكبرى تستثمر مليارات الدولارات في هذا المجال المتنامي.
المملكة العربية السعودية تقود التحول الرقمي في منطقة الشرق الأوسط.
رؤية 2030 تضع التقنية في صميم خطة التنمية الوطنية الشاملة.
الباحثون يطورون نماذج ذكاء اصطناعي أكثر كفاءة وأقل تكلفة.
التعليم يتحول نحو استخدام التقنيات الذكية في الفصول الدراسية.
الصحة الرقمية تستفيد من الذكاء الاصطناعي في التشخيص المبكر.
النقل الذكي يعتمد على السيارات ذاتية القيادة والأنظمة المتقدمة.
الخلاصة أن الذكاء الاصطناعي أصبح ضرورة وليس رفاهية في عالمنا.
"""

# Get a summary (30% of original by default)
result = summarize(text)
print("=== Summary ===")
print(result.text)
print(f"\nOriginal: {result.original_sentences} sentences")
print(f"Summary: {result.sentence_count} sentences")
print(f"Compression: {result.ratio:.0%}")

# Control the compression ratio
short = summarize(text, ratio=0.2)
print(f"\n=== Shorter (20%) === ")
print(short.text)

# Limit max sentences
limited = summarize(text, max_sentences=2)
print(f"\n=== Max 2 sentences ===")
print(limited.text)

# ── 2. Score all sentences ──────────────────────────────────────

print("\n=== Sentence Scores ===")
scored = score_sentences(text, title="الذكاء الاصطناعي")
for s in scored[:5]:
    print(f"  [{s.score:.3f}] {s.text[:60]}...")
    if s.features:
        print(f"         TextRank={s.features['textrank']:.2f}  Position={s.features['position']:.2f}")

# ── 3. Evaluate with ROUGE ──────────────────────────────────────

reference = "الذكاء الاصطناعي يغير العالم. المملكة تقود التحول الرقمي."
generated = result.text

scores = evaluate(reference, generated)
print(f"\n=== ROUGE Evaluation ===")
print(f"ROUGE-1 F1: {scores.rouge1.f1:.4f}")
print(f"ROUGE-2 F1: {scores.rouge2.f1:.4f}")
print(f"ROUGE-L F1: {scores.rougeL.f1:.4f}")
