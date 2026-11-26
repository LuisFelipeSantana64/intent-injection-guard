"""Fine-tune microsoft/deberta-v3-small for 4-class intent and prompt-injection detection."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    ID2LABEL,
    LABEL2ID,
    compute_f1_scores,
    get_env_var,
    load_environment,
    load_labeled_csv,
)

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-small"


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments after loading environment variables."""
    load_environment()

    parser = argparse.ArgumentParser(
        description="Fine-tune microsoft/deberta-v3-small on 4-class support & injection data."
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/train.csv",
        help="Path to training CSV file (default: data/train.csv).",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/test.csv",
        help="Path to evaluation CSV file (default: data/test.csv).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Base pretrained model name or path (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for checkpoints and training logs (default: results).",
    )
    parser.add_argument(
        "--best-model-dir",
        type=str,
        default="./results/best_model",
        help="Directory where the best model, tokenizer, and config are saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs (default: 4).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per device for training and evaluation (default: 16).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW peak learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient (default: 0.01).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence token length (default: 512).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=get_env_var("HF_REPO_ID"),
        help="Hugging Face Hub repo ID (defaults to HF_REPO_ID env var).",
    )
    return parser.parse_args()


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Compute macro and weighted F1 metrics from Trainer evaluation predictions."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    return compute_f1_scores(labels.tolist(), predictions.tolist())


def _tokenize_split(
    csv_path: Path,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Load a CSV split and tokenize text samples into a Hugging Face Dataset."""
    df = load_labeled_csv(csv_path)
    dataset = Dataset.from_pandas(df, preserve_index=False)

    def _tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(_tokenize_batch, batched=True, remove_columns=["text"])


def _build_training_args(args: argparse.Namespace) -> TrainingArguments:
    """Build TrainingArguments with compatibility across transformers v4 and v5."""
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_strategy": "epoch",
        "report_to": "none",
        "seed": args.seed,
    }
    supported_params = inspect.signature(TrainingArguments.__init__).parameters
    eval_param = "eval_strategy" if "eval_strategy" in supported_params else "evaluation_strategy"
    kwargs[eval_param] = "epoch"
    return TrainingArguments(**kwargs)


def _build_trainer(
    model: AutoModelForSequenceClassification,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer: AutoTokenizer,
) -> Trainer:
    """Instantiate Hugging Face Trainer across transformers v4 and v5 signatures."""
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
    }
    supported_params = inspect.signature(Trainer.__init__).parameters
    tok_param = "processing_class" if "processing_class" in supported_params else "tokenizer"
    kwargs[tok_param] = tokenizer
    return Trainer(**kwargs)


def _push_artifacts_to_hub(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    hub_repo_id: str | None,
) -> None:
    """Upload best model, config, and tokenizer to Hugging Face Hub if credentials exist."""
    hf_token = get_env_var("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN is not configured. Skipping upload to Hugging Face Hub.")
        return

    if not hub_repo_id:
        print(
            "[WARNING] HF_TOKEN is set, but HF_REPO_ID (--hub-repo-id) is missing. "
            "Skipping upload to Hugging Face Hub."
        )
        return

    print(f"Uploading best model, config, and tokenizer to Hub repository: {hub_repo_id}")
    trainer.model.push_to_hub(repo_id=hub_repo_id, token=hf_token)
    tokenizer.push_to_hub(repo_id=hub_repo_id, token=hf_token)
    print(f"Successfully pushed model artifacts to https://huggingface.co/{hub_repo_id}")


def main() -> None:
    """Run fine-tuning, evaluation, local best-model export, and optional Hub upload."""
    args = parse_args()
    set_seed(args.seed)

    train_csv = Path(args.train_csv)
    test_csv = Path(args.test_csv)
    best_model_dir = Path(args.best_model_dir)

    print(f"Loading tokenizer and base model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print(f"Tokenizing train split ({train_csv}) and test split ({test_csv})...")
    train_dataset = _tokenize_split(train_csv, tokenizer, args.max_length)
    eval_dataset = _tokenize_split(test_csv, tokenizer, args.max_length)

    training_args = _build_training_args(args)
    trainer = _build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("Starting fine-tuning...")
    trainer.train()

    eval_metrics = trainer.evaluate()
    print(
        f"Best checkpoint evaluation -> "
        f"f1_macro: {eval_metrics.get('eval_f1_macro', 0.0):.4f}, "
        f"f1_weighted: {eval_metrics.get('eval_f1_weighted', 0.0):.4f}"
    )

    best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    print(f"Saved best model, config, and tokenizer locally to: {best_model_dir}")

    _push_artifacts_to_hub(trainer, tokenizer, args.hub_repo_id)


if __name__ == "__main__":
    main()
