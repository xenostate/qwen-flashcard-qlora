#!/usr/bin/env python3
"""
Run every non-empty paragraph from a Markdown/text file through an MLX-LM model
and save the model outputs to a JSON file.

Example:
    python3 scripts/run_gold_set.py \
      --model "$MODEL_PATH" \
      --adapter ./adapters/flashcards-v1-best/0000700_adapters.safetensors \
      --input baselines/gold_set_clean.md \
      --output baselines/adapter_700_outputs.json

To test the base model, omit --adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from mlx_lm import generate, load


SYSTEM_PROMPT = """You convert one supplied source passage into high-quality study flashcards.

Rules:
- You will receive exactly one source passage per request.
- Use only information supported by that source passage.
- Do not use outside knowledge.
- Copy the source passage into the output exactly as provided.
- Do not summarize, rewrite, correct, or shorten the source.
- Each flashcard must test one clear concept.
- Questions must be understandable without seeing the source.
- Answers must be concise but complete.
- Preserve important qualifications, conditions, and exceptions.
- Do not create duplicate or substantially overlapping flashcards.
- If the source contains no useful factual information, return an empty flashcards array.
- Return exactly one valid JSON object.
- Do not include markdown, explanations, comments, or text outside the JSON.

Required schema:
{
  "source": "exact source passage provided by the user",
  "flashcards": [
    {
      "question": "string",
      "answer": "string"
    }
  ]
}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run source passages through an MLX-LM model and save JSON results."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Local MLX model directory or Hugging Face MLX model ID.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter directory or .safetensors checkpoint.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Markdown/text file containing one source passage per paragraph.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON file where all results will be saved.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="Maximum generated tokens per source passage. Default: 700.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="1-based passage index to start from. Useful for resuming manually.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of passages to process.",
    )
    return parser.parse_args()


def read_passages(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()

    # The supplied gold file has one passage per paragraph, separated by blank lines.
    passages = [
        " ".join(paragraph.split())
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if not passages:
        raise ValueError(f"No non-empty passages found in {path}")

    return passages


def strip_code_fence(text: str) -> str:
    """Remove a surrounding Markdown JSON code fence if the model adds one."""
    stripped = text.strip()

    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return stripped


def validate_parsed_output(parsed: Any, expected_source: str) -> list[str]:
    errors: list[str] = []

    if not isinstance(parsed, dict):
        return ["Top-level output is not a JSON object."]

    source = parsed.get("source")
    flashcards = parsed.get("flashcards")

    if source != expected_source:
        errors.append("Returned source does not exactly match the input source.")

    if not isinstance(flashcards, list):
        errors.append("'flashcards' is not an array.")
        return errors

    for card_index, card in enumerate(flashcards, start=1):
        if not isinstance(card, dict):
            errors.append(f"Flashcard {card_index} is not an object.")
            continue

        question = card.get("question")
        answer = card.get("answer")

        if not isinstance(question, str) or not question.strip():
            errors.append(f"Flashcard {card_index} has an invalid question.")
        if not isinstance(answer, str) or not answer.strip():
            errors.append(f"Flashcard {card_index} has an invalid answer.")

    return errors


def save_results(path: Path, results: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "results": results,
    }

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    args = parse_args()

    try:
        passages = read_passages(args.input)
    except (OSError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    start_zero_based = max(args.start_index - 1, 0)
    selected = passages[start_zero_based:]

    if args.limit is not None:
        selected = selected[: args.limit]

    print(f"Found {len(passages)} total passages.")
    print(
        f"Processing {len(selected)} passage(s), starting at index "
        f"{start_zero_based + 1}."
    )
    print(f"Loading model: {args.model}")
    if args.adapter:
        print(f"Loading adapter: {args.adapter}")
    else:
        print("No adapter supplied; testing the base model.")

    load_kwargs: dict[str, Any] = {}
    if args.adapter:
        load_kwargs["adapter_path"] = args.adapter

    try:
        model, tokenizer = load(args.model, **load_kwargs)
    except Exception as exc:
        print(f"Model loading failed: {exc}", file=sys.stderr)
        return 1

    metadata = {
        "model": args.model,
        "adapter": args.adapter,
        "input_file": str(args.input),
        "system_prompt": SYSTEM_PROMPT,
        "max_tokens": args.max_tokens,
        "total_passages_in_input": len(passages),
        "start_index": start_zero_based + 1,
        "requested_count": len(selected),
    }

    results: list[dict[str, Any]] = []

    for offset, source in enumerate(selected):
        source_index = start_zero_based + offset + 1
        print(f"\n[{offset + 1}/{len(selected)}] Running source_{source_index:03d}...")

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Source:\n{source}",
            },
        ]

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            started_at = time.perf_counter()
            raw_output = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
            elapsed_seconds = time.perf_counter() - started_at

            cleaned_output = strip_code_fence(raw_output)

            parsed_output: Any = None
            json_error: str | None = None
            validation_errors: list[str] = []

            try:
                parsed_output = json.loads(cleaned_output)
                validation_errors = validate_parsed_output(parsed_output, source)
            except json.JSONDecodeError as exc:
                json_error = (
                    f"line {exc.lineno}, column {exc.colno}: {exc.msg}"
                )

            result = {
                "id": f"source_{source_index:03d}",
                "source": source,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "valid_json": json_error is None,
                "json_error": json_error,
                "schema_valid": json_error is None and not validation_errors,
                "validation_errors": validation_errors,
                "elapsed_seconds": round(elapsed_seconds, 3),
            }

            results.append(result)

            card_count = (
                len(parsed_output.get("flashcards", []))
                if isinstance(parsed_output, dict)
                and isinstance(parsed_output.get("flashcards"), list)
                else "unknown"
            )

            print(
                f"Completed in {elapsed_seconds:.2f}s | "
                f"valid_json={result['valid_json']} | "
                f"schema_valid={result['schema_valid']} | "
                f"flashcards={card_count}"
            )

        except KeyboardInterrupt:
            print("\nInterrupted. Saving completed results before exiting...")
            save_results(args.output, results, metadata)
            return 130
        except Exception as exc:
            print(f"Generation failed for source_{source_index:03d}: {exc}")
            results.append(
                {
                    "id": f"source_{source_index:03d}",
                    "source": source,
                    "raw_output": None,
                    "parsed_output": None,
                    "valid_json": False,
                    "json_error": None,
                    "schema_valid": False,
                    "validation_errors": [],
                    "generation_error": str(exc),
                }
            )

        # Save after every example so progress survives interruption or failure.
        save_results(args.output, results, metadata)

    valid_json_count = sum(result.get("valid_json", False) for result in results)
    schema_valid_count = sum(result.get("schema_valid", False) for result in results)

    metadata["completed_count"] = len(results)
    metadata["valid_json_count"] = valid_json_count
    metadata["schema_valid_count"] = schema_valid_count
    save_results(args.output, results, metadata)

    print("\nFinished.")
    print(f"Saved results to: {args.output}")
    print(f"Valid JSON: {valid_json_count}/{len(results)}")
    print(f"Schema-valid: {schema_valid_count}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
