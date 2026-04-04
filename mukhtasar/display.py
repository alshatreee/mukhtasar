"""Rich terminal display for mukhtasar."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mukhtasar.summarizer import ScoredSentence, Summary

console = Console()


def display_summary(result: Summary, show_stats: bool = True) -> None:
    """Display summarization result."""
    console.print()

    # Summary text
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


def display_scores(scored: list[ScoredSentence], top_n: int = 10) -> None:
    """Display scored sentences in a table."""
    console.print()

    table = Table(title="[green]Sentence Scores[/green]", border_style="dim")
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Sentence")

    for s in scored[:top_n]:
        # Truncate long sentences
        text = s.text[:120] + "..." if len(s.text) > 120 else s.text
        table.add_row(str(s.index + 1), f"{s.score:.4f}", text)

    console.print(table)
    console.print()


def display_json(result: Summary) -> None:
    """Output summary as JSON."""
    import json
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


def display_explain() -> None:
    """Explain how mukhtasar works."""
    console.print()
    console.print(Panel(
        "[bold green]mukhtasar (مختصر)[/bold green] — Arabic Extractive Summarizer\n\n"
        "[bold]How it works:[/bold]\n\n"
        "1. [green]Split[/green] — Arabic-aware sentence splitting (handles ، ؛ ؟ and mixed text)\n"
        "2. [green]Tokenize[/green] — Remove 220+ Arabic stopwords, normalize alef/teh/yeh variants\n"
        "3. [green]TF-IDF[/green] — Weight each word by importance across all sentences\n"
        "4. [green]Similarity[/green] — Cosine similarity between every sentence pair\n"
        "5. [green]TextRank[/green] — PageRank-style algorithm ranks sentences by centrality\n"
        "6. [green]Select[/green] — Top-ranked sentences returned in original order\n\n"
        "[bold]No API. No model download. No internet. Pure algorithm.[/bold]\n\n"
        "[bold]Dialect support:[/bold] Stopwords cover MSA, Gulf, Egyptian, and Levantine.\n"
        "[bold]Mixed text:[/bold] Handles Arabic + English in the same document.\n"
        "[bold]Formats:[/bold] Plain text, .txt, .jsonl (extracts text/content/body fields).",
        title="[green]Explain[/green]",
        border_style="green",
        padding=(1, 2),
    ))
    console.print()
