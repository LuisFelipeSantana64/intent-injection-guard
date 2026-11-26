"""Dataset assembly, cleaning, class-3 balancing, and stratified 80/20 splitting."""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ID2LABEL, get_env_var, load_environment

CUSTOM_ALLOWED_LABELS = {0, 1, 2}
PROMPT_INJECTION_LABEL = 3
PROMPT_INJECTION_DATASET = "deepset/prompt-injections"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments after loading environment variables."""
    load_environment()
    default_input_csv = get_env_var("CUSTOM_DATA_PATH", "data/support_intents.csv")

    parser = argparse.ArgumentParser(
        description="Build train/test datasets combining custom support intents and deepset/prompt-injections."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=default_input_csv,
        help="Path to CSV with 'text' and 'label' columns (labels 0, 1, 2).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory where train.csv and test.csv will be written.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for the test split (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling and stratified splitting (default: 42).",
    )
    return parser.parse_args()


def _normalize_text(value: object) -> str:
    """Collapse excessive whitespace and trim surrounding spaces."""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clean_dataframe(
    df: pd.DataFrame,
    allowed_labels: set[int],
    source_name: str,
) -> pd.DataFrame:
    """Validate schema, normalize text, drop nulls/duplicates, and validate labels."""
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{source_name}' is missing required columns: {sorted(missing)}."
        )

    cleaned = df[["text", "label"]].copy()
    cleaned = cleaned.dropna(subset=["text", "label"])
    cleaned["text"] = cleaned["text"].map(_normalize_text)
    cleaned = cleaned[cleaned["text"] != ""]

    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="coerce")
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)

    found_labels = set(cleaned["label"].unique().tolist())
    invalid_labels = found_labels - allowed_labels
    if invalid_labels:
        raise ValueError(
            f"Invalid label(s) {sorted(invalid_labels)} in '{source_name}'. "
            f"Expected a subset of {sorted(allowed_labels)}."
        )

    cleaned = cleaned.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    return cleaned


def load_custom_dataset(input_csv: Path) -> pd.DataFrame:
    """Load and clean the user-supplied support intents CSV (classes 0, 1, 2)."""
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV '{input_csv}' was not found. Provide --input-csv, set "
            "CUSTOM_DATA_PATH in .env, or use data/sample_intents.csv as a format template."
        )

    raw_df = pd.read_csv(input_csv)
    cleaned_df = _clean_dataframe(
        raw_df,
        allowed_labels=CUSTOM_ALLOWED_LABELS,
        source_name=str(input_csv),
    )
    if cleaned_df.empty:
        raise ValueError(f"No valid rows remained in '{input_csv}' after cleaning.")
    return cleaned_df


def load_prompt_injections() -> pd.DataFrame:
    """Load deepset/prompt-injections, keep label == 1, and remap to label == 3."""
    dataset = load_dataset(PROMPT_INJECTION_DATASET)
    if isinstance(dataset, DatasetDict):
        combined_hf: Dataset = concatenate_datasets(list(dataset.values()))
    else:
        combined_hf = dataset

    raw_df = combined_hf.to_pandas()
    injection_df = raw_df.loc[raw_df["label"] == 1, ["text", "label"]].copy()
    injection_df["label"] = PROMPT_INJECTION_LABEL

    cleaned_injections = _clean_dataframe(
        injection_df,
        allowed_labels={PROMPT_INJECTION_LABEL},
        source_name=PROMPT_INJECTION_DATASET,
    )
    if cleaned_injections.empty:
        raise ValueError(
            f"No prompt-injection rows (label == 1) found in '{PROMPT_INJECTION_DATASET}'."
        )
    return cleaned_injections


def balance_and_combine(
    custom_df: pd.DataFrame,
    injection_df: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Balance class 3 to match the average volume of classes 0-2 and combine."""
    custom_counts = custom_df["label"].value_counts()
    target_count = max(1, int(round(float(custom_counts.mean()))))

    existing_texts = set(custom_df["text"])
    unique_injections = injection_df[~injection_df["text"].isin(existing_texts)]

    if len(unique_injections) < target_count:
        print(
            f"[WARNING] Requested {target_count} prompt_injection samples to match "
            f"classes 0-2, but only {len(unique_injections)} unique samples are "
            "available in deepset/prompt-injections. Sampling all available without "
            "replacement to prevent train/test data leakage."
        )
        sample_count = len(unique_injections)
    else:
        sample_count = target_count

    sampled_injections = unique_injections.sample(
        n=sample_count,
        replace=False,
        random_state=seed,
    )

    combined = pd.concat([custom_df, sampled_injections], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perform a deterministic 80/20 stratified train/test split."""
    n_classes = int(df["label"].nunique())
    min_class_count = int(df["label"].value_counts().min())
    expected_test_count = math.ceil(len(df) * test_size)

    if min_class_count >= 2 and expected_test_count >= n_classes:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["label"],
            random_state=seed,
        )
    elif min_class_count >= 2 and len(df) >= 2 * n_classes:
        print(
            f"[WARNING] Dataset size ({len(df)}) is too small for stratified "
            f"test_size={test_size} across {n_classes} classes. Adjusting test split "
            f"to {n_classes} samples to preserve stratification."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=n_classes,
            stratify=df["label"],
            random_state=seed,
        )
    else:
        print(
            "[WARNING] Insufficient samples per class for stratified splitting. "
            "Falling back to unstratified train/test split."
        )
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
        )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _format_distribution(df: pd.DataFrame) -> str:
    """Format per-class sample counts for logging."""
    counts = df["label"].value_counts().sort_index()
    parts = [
        f"{label} ({ID2LABEL.get(int(label), 'unknown')}): {count}"
        for label, count in counts.items()
    ]
    return ", ".join(parts)


def main() -> None:
    """Build, balance, split, and save train.csv and test.csv."""
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading custom support dataset from: {input_csv}")
    custom_df = load_custom_dataset(input_csv)
    print(f"Custom dataset loaded ({len(custom_df)} rows): {_format_distribution(custom_df)}")

    print(f"Loading prompt injections from: {PROMPT_INJECTION_DATASET}")
    injection_df = load_prompt_injections()
    print(f"Available unique prompt injections (label=3): {len(injection_df)}")

    combined_df = balance_and_combine(custom_df, injection_df, seed=args.seed)
    print(f"Combined dataset ({len(combined_df)} rows): {_format_distribution(combined_df)}")

    train_df, test_df = split_dataset(combined_df, test_size=args.test_size, seed=args.seed)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved train split ({len(train_df)} rows) -> {train_path}")
    print(f"  Train distribution: {_format_distribution(train_df)}")
    print(f"Saved test split  ({len(test_df)} rows) -> {test_path}")
    print(f"  Test distribution:  {_format_distribution(test_df)}")


if __name__ == "__main__":
    main()
