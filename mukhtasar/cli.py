"""CLI entry point for mukhtasar."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from mukhtasar.display import (
    console,
    display_explain,
    display_json,
    display_rouge,
    display_rouge_json,
    display_scores,
    display_summary,
)
from mukhtasar.rouge import evaluate
from mukhtasar.summarizer import score_sentences, summarize, summarize_file, summarize_multi
from mukhtasar.web import summarize_url


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mukhtasar",
        description="مختصر — Arabic text summarizer. Extractive, offline, multi-feature, dialect-aware.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── text ──
    p = sub.add_parser("text", help="Summarize text from argument or stdin")
    p.add_argument("input", nargs="?", help="Text to summarize (or pipe via stdin)")
    p.add_argument("-r", "--ratio", type=float, default=0.3, help="Summary ratio (default: 0.3)")
    p.add_argument("-n", "--max-sentences", type=int, help="Max sentences")
    p.add_argument("-t", "--title", help="Document title (improves scoring)")
    p.add_argument("--json", action="store_true", help="JSON output")

    # ── file ──
    p = sub.add_parser("file", help="Summarize a file (.txt, .jsonl)")
    p.add_argument("path", help="File path")
    p.add_argument("-r", "--ratio", type=float, default=0.3)
    p.add_argument("-n", "--max-sentences", type=int)
    p.add_argument("-t", "--title", help="Document title")
    p.add_argument("--json", action="store_true")

    # ── multi ──
    p = sub.add_parser("multi", help="Summarize multiple documents with redundancy removal")
    p.add_argument("paths", nargs="+", help="File paths")
    p.add_argument("-r", "--ratio", type=float, default=0.3)
    p.add_argument("-n", "--max-sentences", type=int)
    p.add_argument("--json", action="store_true")

    # ── url ──
    p = sub.add_parser("url", help="Fetch and summarize a web page")
    p.add_argument("url", help="URL to fetch and summarize")
    p.add_argument("-r", "--ratio", type=float, default=0.3)
    p.add_argument("-n", "--max-sentences", type=int)
    p.add_argument("--timeout", type=int, default=15, help="Fetch timeout in seconds")
    p.add_argument("--json", action="store_true")

    # ── score ──
    p = sub.add_parser("score", help="Show sentences ranked by importance")
    p.add_argument("input", nargs="?", help="Text (or pipe via stdin)")
    p.add_argument("-t", "--title", help="Document title")
    p.add_argument("--top", type=int, default=10, help="Show top N")
    p.add_argument("--features", action="store_true", help="Show feature breakdown")

    # ── eval ──
    p = sub.add_parser("eval", help="ROUGE evaluation against a reference summary")
    p.add_argument("--reference", required=True, help="Reference summary file")
    p.add_argument("--summary", required=True, help="Generated summary file")
    p.add_argument("--no-stems", action="store_true", help="Disable Arabic stemming in ROUGE")
    p.add_argument("--json", action="store_true")

    # ── explain ──
    sub.add_parser("explain", help="How mukhtasar works")

    return parser


def _read_input(args_input: str | None) -> str:
    if args_input:
        return args_input
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[red]Error:[/red] No input. Provide text or pipe via stdin.")
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
        result = summarize(text, ratio=args.ratio, max_sentences=args.max_sentences, title=args.title)
        display_json(result) if args.json else display_summary(result)
        return

    if args.command == "file":
        result = summarize_file(args.path, ratio=args.ratio, max_sentences=args.max_sentences, title=args.title)
        display_json(result) if args.json else display_summary(result)
        return

    if args.command == "multi":
        result = summarize_multi(args.paths, ratio=args.ratio, max_sentences=args.max_sentences)
        display_json(result) if args.json else display_summary(result)
        return

    if args.command == "url":
        try:
            result = summarize_url(args.url, ratio=args.ratio, max_sentences=args.max_sentences, timeout=args.timeout)
            display_json(result) if args.json else display_summary(result)
        except (ConnectionError, ValueError) as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        return

    if args.command == "score":
        text = _read_input(args.input)
        scored = score_sentences(text, title=args.title)
        display_scores(scored, top_n=args.top, show_features=args.features)
        return

    if args.command == "eval":
        ref_text = Path(args.reference).read_text(encoding="utf-8")
        sum_text = Path(args.summary).read_text(encoding="utf-8")
        scores = evaluate(ref_text, sum_text, use_stems=not args.no_stems)
        display_rouge_json(scores) if args.json else display_rouge(scores)
        return
