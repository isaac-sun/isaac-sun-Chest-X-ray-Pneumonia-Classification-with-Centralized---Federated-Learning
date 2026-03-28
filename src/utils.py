import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def set_seed(seed: int) -> None:
    """Set random seed for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict:
    """Load YAML config file into a dictionary."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    """Create directory path if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Return CUDA device when available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(data: Dict, path: str) -> None:
    """Save dictionary as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Move all tensors in a state dict to CPU for safe serialization/aggregation."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def compute_binary_metrics(y_true: List[int], y_prob: List[float], threshold: float = 0.5) -> Dict:
    """Compute binary classification metrics from true labels and positive-class probabilities."""
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob_np >= threshold).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred)),
        "precision": float(precision_score(y_true_np, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_np, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred).tolist(),
    }


def epoch_binary_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute batch binary accuracy using sigmoid-thresholded logits."""
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == labels).float().mean().item()


def resolve_path(project_root: Path, path_value: str) -> Path:
    """Resolve relative config path against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def summarize_history(history: Dict[str, List[float]]) -> Dict[str, float]:
    """Extract final scalar values from training history for logging."""
    summary = {}
    for key, values in history.items():
        if values:
            summary[f"final_{key}"] = float(values[-1])
    return summary
