"""Shared configuration, environment loading, model resolution, and metric utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

from dotenv import load_dotenv

if TYPE_CHECKING:
    import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOCAL_MODEL_PATH = "./results/best_model"
DEFAULT_HF_REPO_ID = "Felipe64Sant/intent-injection-guard"

ID2LABEL: dict[int, str] = {
    0: "technical_support",
    1: "general_inquiry",
    2: "urgent_escalation",
    3: "prompt_injection",
}
LABEL2ID: dict[str, int] = {label: idx for idx, label in ID2LABEL.items()}
LABEL_IDS: list[int] = sorted(ID2LABEL.keys())
LABEL_NAMES: list[str] = [ID2LABEL[idx] for idx in LABEL_IDS]


class ModelLoadError(RuntimeError):
    """Raised when loading a local or Hugging Face Hub model fails."""


def load_environment() -> None:
    """Load environment variables from .env in the project root or working directory."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        load_dotenv()


def get_env_var(name: str, default: str | None = None) -> str | None:
    """Resolve an environment variable with Google Colab userdata fallback."""
    value = os.getenv(name)
    if value is not None and value.strip() != "":
        return value.strip()

    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        colab_value = userdata.get(name)
        if colab_value is not None and str(colab_value).strip() != "":
            return str(colab_value).strip()
    except Exception:
        # Either not running on Google Colab or the secret is unavailable.
        pass

    return default


def resolve_model_source(
    explicit_source: str | None = None,
    default_local_path: str = DEFAULT_LOCAL_MODEL_PATH,
    default_hf_repo_id: str = DEFAULT_HF_REPO_ID,
) -> tuple[str, bool]:
    """Resolve local model directory or fall back to Hugging Face Hub repo ID.

    Priority:
    1. Explicit non-default CLI argument (local directory if it exists, or Hub ID).
    2. Local ``./results/best_model`` directory if present.
    3. ``HF_REPO_ID`` environment variable, or ``Felipe64Sant/intent-injection-guard``.
    """
    candidate = explicit_source.strip() if explicit_source and explicit_source.strip() else None

    if candidate and Path(candidate).exists():
        return str(Path(candidate)), True

    looks_like_hub_id = bool(
        candidate
        and "/" in candidate
        and not candidate.startswith((".", "/", "\\"))
        and ":" not in candidate
    )
    is_default_local_arg = candidate is None or candidate == default_local_path

    if not is_default_local_arg:
        if looks_like_hub_id and candidate is not None:
            return candidate, False
        raise ModelLoadError(
            f"Local model path '{candidate}' does not exist. "
            f"Pass a valid directory, a Hugging Face repo ID (e.g. '{default_hf_repo_id}'), "
            "or omit the argument to use automatic fallback."
        )

    local_default = Path(default_local_path)
    if local_default.exists():
        return str(local_default), True

    hub_repo = get_env_var("HF_REPO_ID") or default_hf_repo_id
    print(
        f"Local model '{default_local_path}' not found. "
        f"Falling back to Hugging Face Hub: '{hub_repo}'."
    )
    return hub_repo, False


def _iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield an exception and all chained causes/contexts without cycles."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def format_model_load_error(
    model_source: str,
    is_local: bool,
    has_token: bool,
    exc: BaseException,
) -> str:
    """Translate low-level Hugging Face / network exceptions into actionable messages."""
    if is_local:
        return (
            f"Failed to load local model from '{model_source}' ({type(exc).__name__}: {exc}). "
            "Verify the directory contains valid model weights, config.json, and tokenizer files, "
            "or run 'python src/train.py' to regenerate it."
        )

    chain = list(_iter_exception_chain(exc))
    combined_msg = " ".join(str(err).lower() for err in chain)
    class_names = {type(err).__name__ for err in chain}
    status_codes = {
        getattr(getattr(err, "response", None), "status_code", None) for err in chain
    }

    auth_classes = {"GatedRepoError", "LocalTokenNotFoundError", "RepositoryNotFoundError"}
    is_auth_or_missing_repo = (
        bool(class_names & auth_classes)
        or bool(status_codes & {401, 403})
        or any(
            token in combined_msg
            for token in ("401", "403", "unauthorized", "gated repo", "access token", "invalid token")
        )
    )
    if is_auth_or_missing_repo:
        if not has_token:
            return (
                f"Failed to load '{model_source}' from Hugging Face Hub: authentication is required "
                "or the repository is private/unavailable and HF_TOKEN is not set. "
                "Add HF_TOKEN to your .env file (or Colab Secrets), or train locally with "
                "'python src/train.py'."
            )
        return (
            f"Failed to load '{model_source}' from Hugging Face Hub with the configured HF_TOKEN "
            "(401/403 or repository not found). Verify that HF_TOKEN is valid and has read access "
            f"to '{model_source}'."
        )

    network_classes = {
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
        "ConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "TimeoutError",
        "NewConnectionError",
        "NameResolutionError",
    }
    is_connection_error = (
        bool(class_names & network_classes)
        or any(
            token in combined_msg
            for token in (
                "couldn't connect",
                "could not connect",
                "connection",
                "offline",
                "timed out",
                "name resolution",
                "getaddrinfo",
                "network",
            )
        )
    )
    if is_connection_error:
        return (
            f"Connection error while downloading '{model_source}' from Hugging Face Hub. "
            "Check your internet connection/proxy settings, ensure the model is cached locally, "
            "or run 'python src/train.py' to create ./results/best_model."
        )

    return (
        f"Failed to load model '{model_source}' from Hugging Face Hub "
        f"({type(exc).__name__}: {exc}). Check your internet connection and HF_TOKEN."
    )


def load_labeled_csv(
    csv_path: Path,
    allowed_labels: set[int] | None = None,
) -> pd.DataFrame:
    """Load and validate a labeled CSV containing 'text' and integer 'label' columns."""
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: '{csv_path}'.")

    df = pd.read_csv(csv_path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required columns: {sorted(missing)}."
        )

    cleaned = df[["text", "label"]].copy()
    cleaned = cleaned.dropna(subset=["text", "label"])
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"] != ""]
    cleaned["label"] = pd.to_numeric(cleaned["label"], errors="coerce")
    cleaned = cleaned.dropna(subset=["label"])
    cleaned["label"] = cleaned["label"].astype(int)

    valid_labels = allowed_labels if allowed_labels is not None else set(LABEL_IDS)
    invalid = set(cleaned["label"].unique().tolist()) - valid_labels
    if invalid:
        raise ValueError(
            f"Invalid label(s) {sorted(invalid)} in '{csv_path}'. "
            f"Expected subset of {sorted(valid_labels)}."
        )

    if cleaned.empty:
        raise ValueError(f"CSV '{csv_path}' contains no valid rows after cleaning.")

    return cleaned.reset_index(drop=True)


def compute_f1_scores(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute macro and weighted F1 scores for multi-class predictions."""
    from sklearn.metrics import f1_score

    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
