"""CLI entry point for mukhtasar."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn

from mukhtasar.display import console, display_explain, display_json, display_scores, display_summary
from mukhtasar.summarizer import score_sentences, summarize, summarize_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mukhtasar",
        description="مختصر — Arabic text summarizer. Extractive, offline, dialect-aware.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── summarize (text) ──
    p_text = sub.add_parser("text", help="Summarize text from argument or stdin")
    p_text.add_argument("input", nargs="?", help="Text to summarize (or pipe via stdin)")
    p_text.add_argument("-r", "--ratio", type=float, default=0.3, help="Summary ratio (default: 0.3)")
    p_text.add_argument("-n", "--max-sentences", type=int, help="Max sentences in summary")
    p_text.add_argument("--json", action="store_true", help="Output as JSON")

    # ── summarize (file) ──
    p_file = sub.add_parser("file", help="Summarize text from a file (.txt, .jsonl)")
    p_file.add_argument("path", help="File path")
    p_file.add_argument("-r", "--ratio", type=float, default=0.3, help="Summary ratio (default: 0.3)")
    p_file.add_argument("-n", "--max-sentences", type=int, help="Max sentences in summary")
    p_file.add_argument("--json", action="store_true", help="Output as JSON")

    # ── score ──
    p_score = sub.add_parser("score", help="Show all sentences ranked by importance")
    p_score.add_argument("input", nargs="?", help="Text to score (or pipe via stdin)")
    p_score.add_argument("-t", "--top", type=int, default=10, help="Show top N sentences")

    # ── explain ──
    sub.add_parser("explain", help="How mukhtasar works")

    return parser


def _read_input(args_input: str | None) -> str:
    """Read text from argument or stdin."""
    if args_input:
        return args_input
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[red]Error:[/red] No input. Provide text as argument or pipe via stdin.")
    sys.exit(1)


def main() -> NoReturn | None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "explain":
        display_explain()
        return

    if args.command == "text":
        text = _read_input(args.input)
        result = summarize(text, ratio=args.ratio, max_sentences=args.max_sentences)
        if args.json:
            display_json(result)
        else:
            display_summary(result)
        return

    if args.command == "file":
        result = summarize_file(args.path, ratio=args.ratio, max_sentences=args.max_sentences)
        if args.json:
            display_json(result)
        else:
            display_summary(result)
        return

    if args.command == "score":
        text = _read_input(args.input)
        scored = score_sentences(text)
        display_scores(scored, top_n=args.top)
        return
