#!/usr/bin/env python3

import argparse
import json
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


SOURCES = [
    "Dynamic programming solves problems with overlapping subproblems by storing previously computed solutions and reusing them instead of recomputing them.",
    "A Support Vector Machine (SVM) is a supervised learning algorithm that outputs an optimal hyperplane which categorizes new examples in classification tasks.",
    "Time complexity, often expressed using Big O notation, quantifies the amount of time taken by an algorithm to run as a function of the length of the input.",
    "A deadlock is a situation in computing where two or more competing actions are each waiting for the other to finish, preventing any of them from resolving.",
    "Random Forest is an ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the mode of their classes.",
    "A Data Lake is a centralized repository that allows you to store all your structured and unstructured data at any scale in its native format.",
    "Convolutional Neural Networks (CNNs) are a class of deep neural networks heavily used in computer vision tasks because they can capture spatial hierarchies in images.",
    "Representational State Transfer (REST) is an architectural style for networked applications whose constraints include client-server separation and stateless communication.",
    "L2 regularization (Ridge) adds a penalty equal to the square of the magnitude of coefficients to the loss function, which helps to prevent overfitting by penalizing large weights.",
    "MapReduce is a programming model for processing and generating large data sets with a parallel, distributed algorithm on a cluster.",
    "A low-mass star spends most of its life fusing hydrogen in its core. After the core hydrogen is depleted, the star expands into a red giant. It later sheds its outer layers, leaving behind a dense white dwarf. The color of the telescope used to observe the star does not determine these stages.",
    "A catalyst lowers the activation energy of both the forward and reverse reactions. It can help a reversible reaction reach equilibrium faster, but it does not change the equilibrium constant or the final equilibrium composition at a fixed temperature.",
    "Backpropagation applies the chain rule from the output layer toward earlier layers. At each layer, the gradient arriving from later computations is multiplied by the local derivative. The resulting gradients indicate how a small change in each parameter would affect the loss. The optimizer, not backpropagation itself, uses those gradients to update the parameters.",
    "The document contains a heading titled 'Results', followed by a blank table and the note, 'Values will be inserted after the experiment is completed.'",
    "The printing press made it cheaper and faster to reproduce written works in Europe. This supported wider circulation of religious, scientific, and political ideas. However, increased circulation did not mean that everyone immediately became literate or gained equal access to books.",
    "Demand-pull inflation can occur when total demand grows faster than an economy's ability to produce goods and services. Cost-push inflation can occur when production costs rise and firms pass some of those costs to consumers. A single period of higher prices does not by itself identify which mechanism caused the increase.",
    "During gene expression, DNA is transcribed into messenger RNA in the nucleus of a eukaryotic cell. The messenger RNA is then processed and transported to the cytoplasm, where ribosomes translate its sequence into a protein. Not every region of DNA is expressed in every cell.",
    "In copy-on-write memory, two processes may initially share the same physical memory pages after a fork. The operating system marks those pages so that if either process tries to modify one, a private copy is created for that process. Pages that are never modified can remain shared, reducing unnecessary copying.",
    "A correlation between two variables shows that they vary together, but it does not by itself prove that one causes the other. A third variable may influence both, or the apparent relationship may result from selection bias or chance. Establishing causation generally requires stronger evidence than observing correlation alone.",
    "In a decoder-only language model, causal masking prevents each token from attending to tokens that come later in the sequence. This preserves the next-token prediction setup during training. The mask does not prevent a token from attending to itself or to earlier tokens. Padding masks solve a different problem by hiding padding positions.",
]


def build_prompt(tokenizer: Any, source: str) -> str:
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

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def validate_response(parsed: Any, expected_source: str, index: int) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError(f"response {index} is not a JSON object")

    source = parsed.get("source")
    flashcards = parsed.get("flashcards")

    if source != expected_source:
        raise ValueError(f"response {index} did not preserve the source exactly")

    if not isinstance(flashcards, list):
        raise ValueError(f"response {index} has a non-array flashcards field")

    for card_index, card in enumerate(flashcards):
        if not isinstance(card, dict):
            raise ValueError(f"response {index}, card {card_index} is not an object")

        question = card.get("question")
        answer = card.get("answer")

        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"response {index}, card {card_index} has an empty question")

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"response {index}, card {card_index} has an empty answer")

    return {
        "source": source,
        "flashcards": flashcards,
    }


def generate_one(model: Any, tokenizer: Any, source: str, max_tokens: int) -> dict[str, Any]:
    prompt = build_prompt(tokenizer, source)
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    return json.loads(response)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prompt an MLX fine-tuned flashcard model with fixed evaluation sources."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mlx_batch_results.json"),
        help="Path to write the single JSON array of model outputs.",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    model, tokenizer = load(
        args.model,
        adapter_path=args.adapter,
    )

    results: list[dict[str, Any]] = []

    for index, source in enumerate(SOURCES, start=1):
        print(f"Generating {index}/{len(SOURCES)}...")
        try:
            parsed = generate_one(model, tokenizer, source, args.max_tokens)
            results.append(validate_response(parsed, source, index))
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Generation failed for source {index}: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
