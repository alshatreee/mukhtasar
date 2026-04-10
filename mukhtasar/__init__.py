"""mukhtasar (مختصر) — Arabic Text Summarizer & NLP Toolkit."""

__version__ = "0.3.0"

__all__ = [
    # Summarization
    "summarize", "summarize_file", "summarize_multi", "summarize_url",
    "score_sentences",
    # Evaluation
    "evaluate",
    # Arabic NLP tools (reusable in other projects like kazima-ai)
    "normalize", "tokenize", "light_stem", "split_sentences",
    "is_arabic", "clean_html",
]

from mukhtasar.arabic import is_arabic, light_stem, normalize, split_sentences, tokenize
from mukhtasar.rouge import evaluate
from mukhtasar.summarizer import score_sentences, summarize, summarize_file, summarize_multi
from mukhtasar.web import clean_html, summarize_url
