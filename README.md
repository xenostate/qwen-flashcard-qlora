# Flashcard Fine-Tuning

This project contains data and utility scripts for fine-tuning and evaluating an MLX language model that converts source passages into study flashcards.

## Layout

- `data/raw/` contains source datasets before train/validation splitting.
- `data/raw/synthetic/` contains the 25 synthetic batches, 20 examples per batch.
- `data/processed/` contains the combined dataset and MLX-ready train/validation splits.
- `scripts/` contains validation, conversion, generation, and evaluation utilities.
- `baselines/` contains prompts, gold-set passages, and saved baseline/evaluation outputs.
- `adapters/` contains local adapter metadata and weights. Weight files are ignored by git by default.

## Dataset

- `data/processed/synthetic_all.json`: one JSON array with 550 examples.
- `data/processed/train.json`: 494 training examples.
- `data/processed/validate.json`: 56 validation examples.
- `data/processed/train.jsonl`: MLX chat-format training data.
- `data/processed/valid.jsonl`: MLX chat-format validation data.

## Model

The fine-tuned LoRA adapter is available on Hugging Face:
https://huggingface.co/tenroman/qwen3-vl-8b-flashcard-qlora

## Common Commands

Validate raw synthetic batches:

```bash
python3 scripts/validate_batch_json.py --dry-run data/raw/synthetic/batch_*.json
```

Convert a JSON array to MLX chat JSONL:

```bash
python3 scripts/convert_to_mlx_jsonl.py data/processed/train.json data/processed/train.jsonl
```

Run the gold-set evaluator:

```bash
python3 scripts/run_gold_set.py \
  --model "$MODEL_PATH" \
  --adapter adapters/flashcards-v1-best/adapters.safetensors \
  --input baselines/gold_set_clean.md \
  --output baselines/adapter_outputs.json
```

Run the fixed 20-passage batch generator:

```bash
python3 scripts/generate_mlx_flashcards_batch.py \
  --model "$MODEL_PATH" \
  --adapter adapters/flashcards-v1-best/adapters.safetensors \
  --output baselines/mlx_batch_results.json
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
