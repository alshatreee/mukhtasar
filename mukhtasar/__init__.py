"""mukhtasar (مختصر) — Arabic Text Summarizer."""

__version__ = "0.2.0"

__all__ = ["summarize", "summarize_file", "summarize_multi", "score_sentences", "evaluate"]

from mukhtasar.rouge import evaluate
from mukhtasar.summarizer import score_sentences, summarize, summarize_file, summarize_multi
