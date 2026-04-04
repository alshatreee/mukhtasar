"""Rich terminal display for mukhtasar."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mukhtasar.rouge import RougeScores
from mukhtasar.summarizer import ScoredSentence, Summary

console = Console()


def display_summary(result: Summary, show_stats: bool = True) -> None:
    console.print()
    console.print(Panel(
        Text(result.text, style="bold"),
        title="[green]مختصر — Summary[/green]",
        border_style="green",
        padding=(1, 2),
    ))
    if show_stats:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="dim")
        table.add_column(style="green")
        table.add_row("Original", f"{result.original_sentences} sentences · {result.original_length:,} chars")
        table.add_row("Summary", f"{result.sentence_count} sentences · {result.summary_length:,} chars")
        table.add_row("Compression", f"{result.ratio:.0%} of original")
        console.print(table)
    console.print()


def display_scores(scored: list[ScoredSentence], top_n: int = 10, show_features: bool = False) -> None:
    console.print()
    table = Table(title="[green]Sentence Scores[/green]", border_style="dim")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)

    if show_features:
        table.add_column("TR", style="cyan", width=5)
        table.add_column("Pos", style="cyan", width=5)
        table.add_column("Cue", style="cyan", width=5)
        table.add_column("Num", style="cyan", width=5)
        table.add_column("PN", style="cyan", width=5)

    table.add_column("Sentence")

    for s in scored[:top_n]:
        text = s.text[:100] + "..." if len(s.text) > 100 else s.text
        row = [str(s.index + 1), f"{s.score:.4f}"]

        if show_features and s.features:
            row.extend([
                f"{s.features.get('textrank', 0):.2f}",
                f"{s.features.get('position', 0):.2f}",
                f"{s.features.get('cue_words', 0):.2f}",
                f"{s.features.get('numbers', 0):.2f}",
                f"{s.features.get('proper_nouns', 0):.2f}",
            ])

        row.append(text)
        table.add_row(*row)

    console.print(table)
    console.print()


def display_rouge(scores: RougeScores) -> None:
    console.print()
    table = Table(title="[green]ROUGE Evaluation[/green]", border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Precision", style="cyan", justify="right")
    table.add_column("Recall", style="cyan", justify="right")
    table.add_column("F1", style="green", justify="right")

    table.add_row("ROUGE-1", f"{scores.rouge1.precision:.4f}", f"{scores.rouge1.recall:.4f}", f"{scores.rouge1.f1:.4f}")
    table.add_row("ROUGE-2", f"{scores.rouge2.precision:.4f}", f"{scores.rouge2.recall:.4f}", f"{scores.rouge2.f1:.4f}")
    table.add_row("ROUGE-L", f"{scores.rougeL.precision:.4f}", f"{scores.rougeL.recall:.4f}", f"{scores.rougeL.f1:.4f}")

    console.print(table)
    console.print()


def display_json(result: Summary) -> None:
    output = {
        "summary": result.text,
        "sentences": result.sentences,
        "original_length": result.original_length,
        "summary_length": result.summary_length,
        "compression_ratio": round(result.ratio, 3),
        "sentence_count": result.sentence_count,
        "original_sentences": result.original_sentences,
    }
    console.print(json.dumps(output, ensure_ascii=False, indent=2))


def display_rouge_json(scores: RougeScores) -> None:
    output = {
        "rouge1": {"precision": scores.rouge1.precision, "recall": scores.rouge1.recall, "f1": scores.rouge1.f1},
        "rouge2": {"precision": scores.rouge2.precision, "recall": scores.rouge2.recall, "f1": scores.rouge2.f1},
        "rougeL": {"precision": scores.rougeL.precision, "recall": scores.rougeL.recall, "f1": scores.rougeL.f1},
    }
    console.print(json.dumps(output, indent=2))


def display_explain() -> None:
    console.print()
    console.print(Panel(
        "[bold green]mukhtasar (مختصر)[/bold green] — Arabic Extractive Summarizer\n\n"
        "[bold]Scoring features (7 combined):[/bold]\n\n"
        "  1. [green]TextRank[/green] (35%) — Graph-based sentence centrality via PageRank\n"
        "  2. [green]Position[/green] (15%) — First/last sentences score higher\n"
        "  3. [green]Cue Words[/green] (15%) — Arabic markers: الخلاصة، أهم، النتيجة، بالتالي\n"
        "  4. [green]Title Sim[/green] (15%) — Word overlap with document title\n"
        "  5. [green]Proper Nouns[/green] (10%) — Named entity density\n"
        "  6. [green]Numbers[/green] (5%) — Sentences with statistics/data\n"
        "  7. [green]Length[/green] (5%) — Penalize fragments, favor complete sentences\n\n"
        "[bold]Text processing:[/bold]\n\n"
        "  • 250+ Arabic stopwords (MSA, Gulf, Egyptian, Levantine)\n"
        "  • Light stemming: كتابات→كتاب, المطورون→مطور\n"
        "  • Arabic normalization: alef/teh/yeh unification, diacritic removal\n"
        "  • Smart sentence splitting: handles ، ؛ ؟ bullets, quoted speech\n\n"
        "[bold]Evaluation:[/bold]\n\n"
        "  • ROUGE-1, ROUGE-2, ROUGE-L with Arabic lemma-aware tokenization\n"
        "  • Multi-document summarization with redundancy removal\n\n"
        "[bold]No API. No model. No internet. Pure algorithm.[/bold]",
        title="[green]Explain[/green]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()
