#!/usr/bin/env python3
"""Validate and clean synthetic flashcard batch JSON files.

For each input file, valid examples remain in the original file and rejected
examples are written to a sibling file named <stem>_rejected.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def is_blank(value: Any) -> bool:
    return not isinstance(value, str) or value.strip() == ""


def rejected_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_rejected.json")


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_flashcards(example: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    flashcards = example.get("flashcards")

    if not isinstance(flashcards, list):
        return ["flashcards is not an array"]

    seen_cards: set[tuple[str, str]] = set()
    duplicate_cards: set[tuple[str, str]] = set()

    for index, card in enumerate(flashcards):
        if not isinstance(card, dict):
            errors.append(f"flashcards[{index}] is not an object")
            continue

        question_missing = "question" not in card
        answer_missing = "answer" not in card

        if question_missing:
            errors.append(f"flashcards[{index}].question is missing")
        elif is_blank(card["question"]):
            errors.append(f"flashcards[{index}].question is empty")

        if answer_missing:
            errors.append(f"flashcards[{index}].answer is missing")
        elif is_blank(card["answer"]):
            errors.append(f"flashcards[{index}].answer is empty")

        if not question_missing and not answer_missing:
            question = card["question"]
            answer = card["answer"]
            if isinstance(question, str) and isinstance(answer, str):
                pair = (question, answer)
                if pair in seen_cards:
                    duplicate_cards.add(pair)
                seen_cards.add(pair)

    for question, answer in sorted(duplicate_cards):
        errors.append(f"two cards are exact duplicates: question={question!r}, answer={answer!r}")

    return errors


def validate_example(
    example: Any,
    seen_ids: set[str],
    seen_sources: set[str],
) -> tuple[list[str], str | None, str | None]:
    errors: list[str] = []

    if not isinstance(example, dict):
        return ["example is not an object"], None, None

    example_id = example.get("id")
    source = example.get("source")

    if is_blank(example_id):
        errors.append("id is missing or empty")
        example_id = None
    elif example_id in seen_ids:
        errors.append(f"id is duplicated: {example_id}")

    if is_blank(source):
        errors.append("source is missing or empty")
        source = None
    elif source in seen_sources:
        errors.append("the same source already exists")

    errors.extend(validate_flashcards(example))

    return errors, example_id, source


def rejection_entry(errors: Iterable[str], example: Any) -> dict[str, Any]:
    return {
        "validation_errors": list(errors),
        "example": example,
    }


def validate_file(
    path: Path,
    seen_ids: set[str],
    seen_sources: set[str],
    *,
    dry_run: bool,
) -> tuple[int, int, bool]:
    rejected: list[dict[str, Any]] = []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"{path}: cannot read file: {exc}", file=sys.stderr)
        return 0, 0, True

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        rejected.append(
            {
                "validation_errors": [f"JSON cannot be parsed: {exc.msg} at line {exc.lineno}, column {exc.colno}"],
                "raw": raw,
            }
        )
        if not dry_run:
            dump_json(rejected_path(path), rejected)
        print(f"{path}: rejected whole file because JSON cannot be parsed", file=sys.stderr)
        return 0, 1, True

    if not isinstance(data, list):
        rejected.append(rejection_entry(["top level is not an array"], data))
        if not dry_run:
            dump_json(path, [])
            dump_json(rejected_path(path), rejected)
        print(f"{path}: rejected whole file because top level is not an array", file=sys.stderr)
        return 0, 1, True

    valid_examples: list[Any] = []

    for example in data:
        errors, example_id, source = validate_example(example, seen_ids, seen_sources)

        if errors:
            rejected.append(rejection_entry(errors, example))
            continue

        if example_id is not None:
            seen_ids.add(example_id)
        if source is not None:
            seen_sources.add(source)
        valid_examples.append(example)

    if not dry_run:
        dump_json(path, valid_examples)
        if rejected:
            dump_json(rejected_path(path), rejected)
        else:
            reject_file = rejected_path(path)
            if reject_file.exists():
                reject_file.unlink()

    print(f"{path}: kept {len(valid_examples)}, rejected {len(rejected)}")
    return len(valid_examples), len(rejected), bool(rejected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate flashcard batch JSON files, moving rejected examples to *_rejected.json files.",
    )
    parser.add_argument("files", nargs="+", type=Path, help="One or more JSON batch files to validate.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report validation results without rewriting files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    total_kept = 0
    total_rejected = 0
    had_rejections = False

    for path in args.files:
        kept, rejected, file_had_rejections = validate_file(
            path,
            seen_ids,
            seen_sources,
            dry_run=args.dry_run,
        )
        total_kept += kept
        total_rejected += rejected
        had_rejections = had_rejections or file_had_rejections

    print(f"Total kept {total_kept}, rejected {total_rejected}")
    return 1 if had_rejections else 0


if __name__ == "__main__":
    raise SystemExit(main())
