"""Evaluate the fine-tuned classifier and export a confusion matrix figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    DEFAULT_HF_REPO_ID,
    DEFAULT_LOCAL_MODEL_PATH,
    LABEL_IDS,
    LABEL_NAMES,
    ModelLoadError,
    compute_f1_scores,
    format_model_load_error,
    get_env_var,
    load_environment,
    load_labeled_csv,
    resolve_model_source,
)


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI arguments after loading environment variables."""
    load_environment()

    parser = argparse.ArgumentParser(
        description="Evaluate intent-injection-guard on test.csv and plot confusion matrix."
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/test.csv",
        help="Path to test CSV file (default: data/test.csv).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_LOCAL_MODEL_PATH,
        help=(
            "Local model directory or HF Hub repo ID "
            f"(defaults to {DEFAULT_LOCAL_MODEL_PATH}, falling back to HF_REPO_ID or {DEFAULT_HF_REPO_ID})."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="assets/confusion_matrix.png",
        help="Destination path for the confusion matrix PNG (default: assets/confusion_matrix.png).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence token length (default: 512).",
    )
    return parser.parse_args()


def load_model_and_tokenizer(
    model_path: str,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, str]:
    """Load model and tokenizer from local disk or Hugging Face Hub fallback."""
    model_source, is_local = resolve_model_source(model_path)

    load_kwargs: dict[str, str] = {}
    hf_token = get_env_var("HF_TOKEN")
    if not is_local and hf_token:
        load_kwargs["token"] = hf_token

    print(f"Loading model and tokenizer from: {model_source}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)
        model = AutoModelForSequenceClassification.from_pretrained(model_source, **load_kwargs)
    except Exception as exc:
        raise ModelLoadError(
            format_model_load_error(
                model_source=model_source,
                is_local=is_local,
                has_token=bool(hf_token),
                exc=exc,
            )
        ) from exc

    return model, tokenizer, model_source


def predict_labels(
    texts: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
) -> list[int]:
    """Run batched inference and return predicted integer class IDs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            logits = model(**encoded).logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
            predictions.extend(int(pred) for pred in batch_preds)
    return predictions


def save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_path: Path,
) -> None:
    """Plot and save a formatted confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cbar=True,
        square=True,
        ax=ax,
    )
    ax.set_title("Confusion Matrix — intent-injection-guard", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Load test dataset and model, print evaluation metrics, and plot confusion matrix."""
    args = parse_args()
    test_csv = Path(args.test_csv)
    try:
        test_df = load_labeled_csv(test_csv)
    except FileNotFoundError:
        print(
            f"[ERROR] Test dataset '{test_csv}' not found. "
            "Run 'python data/build_dataset.py' first or pass '--test-csv <path>' "
            "(for example: '--test-csv data/sample_intents.csv').",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        model, tokenizer, _ = load_model_and_tokenizer(args.model_path)
    except ModelLoadError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    y_true = test_df["label"].tolist()
    y_pred = predict_labels(
        texts=test_df["text"].tolist(),
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABEL_IDS,
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        )
    )

    scores = compute_f1_scores(y_true, y_pred)
    print(f"f1_macro:    {scores['f1_macro']:.4f}")
    print(f"f1_weighted: {scores['f1_weighted']:.4f}")

    output_path = Path(args.output_path)
    save_confusion_matrix(y_true, y_pred, output_path)
    print(f"\nConfusion matrix saved to: {output_path}")


if __name__ == "__main__":
    main()
