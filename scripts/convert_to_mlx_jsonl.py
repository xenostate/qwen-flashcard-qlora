#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Any


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


def load_json_array(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not valid JSON: line {exc.lineno}, column {exc.colno}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a top-level JSON array.")

    return data


def validate_example(example: dict[str, Any], index: int) -> None:
    if not isinstance(example, dict):
        raise ValueError(f"Example {index} must be a JSON object.")

    example_id = example.get("id")
    source = example.get("source")
    flashcards = example.get("flashcards")

    if not isinstance(example_id, str) or not example_id.strip():
        raise ValueError(f"Example {index} has a missing or invalid id.")

    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{example_id}: source must be a non-empty string.")

    if not isinstance(flashcards, list):
        raise ValueError(f"{example_id}: flashcards must be an array.")

    for card_index, card in enumerate(flashcards):
        if not isinstance(card, dict):
            raise ValueError(
                f"{example_id}: flashcard {card_index} must be an object."
            )

        question = card.get("question")
        answer = card.get("answer")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"{example_id}: flashcard {card_index} has an invalid question."
            )

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"{example_id}: flashcard {card_index} has an invalid answer."
            )


def convert_example(example: dict[str, Any]) -> dict[str, Any]:
    source = example["source"]

    assistant_output = {
        "source": source,
        "flashcards": example["flashcards"],
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Source:\n{source}",
            },
            {
                "role": "assistant",
                # The assistant completion itself must be JSON text.
                "content": json.dumps(
                    assistant_output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
    }


def convert_file(input_path: Path, output_path: Path) -> None:
    examples = load_json_array(input_path)

    seen_ids: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for index, example in enumerate(examples):
            validate_example(example, index)

            example_id = example["id"]
            if example_id in seen_ids:
                raise ValueError(f"Duplicate ID found: {example_id}")

            seen_ids.add(example_id)

            converted = convert_example(example)

            # One complete JSON object per physical line.
            output_file.write(
                json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
            )
            output_file.write("\n")

    print(f"Converted {len(examples)} examples")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert flashcard JSON arrays into MLX-LM chat JSONL."
    )
    parser.add_argument("input", type=Path, help="Input JSON array")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    args = parser.parse_args()

    try:
        convert_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Conversion failed: {exc}") from exc


if __name__ == "__main__":
    main()