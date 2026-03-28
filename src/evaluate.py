from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, roc_curve

from model import get_target_layer
from model import build_model
from utils import compute_binary_metrics, get_device, load_config, save_json


@torch.no_grad()
def collect_predictions(model, data_loader, device: torch.device) -> Tuple[List[int], List[float]]:
    """Run inference and collect true labels and predicted probabilities."""
    model.eval()
    y_true, y_prob = [], []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images).view(-1)
        probs = torch.sigmoid(logits)

        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.cpu().numpy().astype(float).tolist())

    return y_true, y_prob


def evaluate_model(model, data_loader, device: torch.device) -> Dict:
    """Compute core binary metrics and ROC-AUC for a trained model."""
    y_true, y_prob = collect_predictions(model, data_loader, device)
    metrics = compute_binary_metrics(y_true, y_prob)

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def plot_confusion_matrix(conf_mat: List[List[int]], class_names: List[str], save_path: Path, title: str) -> None:
    """Plot and save confusion matrix heatmap."""
    cm = np.array(conf_mat)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true: List[int], y_prob: List[float], save_path: Path, title: str) -> float:
    """Plot ROC curve and return AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    return float(roc_auc)


def plot_training_comparison(
    centralized_history: Dict[str, List[float]],
    federated_history: Dict[str, List[float]],
    save_dir: Path,
) -> None:
    """Plot centralized vs federated training curves for loss and accuracy."""
    rounds = np.arange(1, len(federated_history.get("val_loss", [])) + 1)
    epochs = np.arange(1, len(centralized_history.get("val_loss", [])) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, centralized_history.get("val_loss", []), marker="o", label="Centralized")
    plt.plot(rounds, federated_history.get("val_loss", []), marker="s", label="Federated")
    plt.xlabel("Epoch / Round")
    plt.ylabel("Validation Loss")
    plt.title("Loss Curve: Centralized vs Federated")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, centralized_history.get("val_accuracy", []), marker="o", label="Centralized")
    plt.plot(rounds, federated_history.get("val_accuracy", []), marker="s", label="Federated")
    plt.xlabel("Epoch / Round")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy Curve: Centralized vs Federated")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve_comparison.png", dpi=200)
    plt.close()


def plot_metrics_bar(central_metrics: Dict, fed_metrics: Dict, save_path: Path) -> None:
    """Create bar chart comparing key metrics of centralized and federated models."""
    metric_names = ["accuracy", "precision", "recall", "f1"]
    c_values = [central_metrics.get(m, 0.0) for m in metric_names]
    f_values = [fed_metrics.get(m, 0.0) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, c_values, width=width, label="Centralized")
    plt.bar(x + width / 2, f_values, width=width, label="Federated")
    plt.xticks(x, [m.upper() for m in metric_names])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Performance Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def denormalize_image(image: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """Convert normalized tensor image to displayable NumPy RGB array."""
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = image.detach().cpu() * std_t + mean_t
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def plot_example_predictions(model, data_loader, device: torch.device, config: Dict, save_path: Path, n: int = 8) -> None:
    """Plot sample predictions with probabilities for qualitative inspection."""
    class_names = getattr(data_loader.dataset, "classes", ["NORMAL", "PNEUMONIA"])
    mean = config["data"]["normalize_mean"]
    std = config["data"]["normalize_std"]

    model.eval()
    images_batch, labels_batch = next(iter(data_loader))
    images_batch = images_batch.to(device)

    with torch.no_grad():
        logits = model(images_batch).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

    images = images_batch.cpu()
    labels = labels_batch.numpy().astype(int)

    n = min(n, len(images))
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(14, 3.5 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        img = denormalize_image(images[i], mean, std)
        pred_label = int(probs[i] >= 0.5)
        true_label = labels[i]
        plt.imshow(img)
        plt.axis("off")
        plt.title(
            f"True: {class_names[true_label]}\\nPred: {class_names[pred_label]} ({probs[i]:.2f})",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


class GradCAM:
    """Minimal Grad-CAM utility for binary classifiers."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate normalized CAM heatmap for one input image tensor."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits.squeeze()
        score.backward(retain_graph=False)

        grads = self.gradients
        acts = self.activations
        if grads is None or acts is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations.")

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def close(self):
        """Remove hooks to avoid leaking references."""
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def plot_gradcam_examples(model, model_name: str, data_loader, device: torch.device, config: Dict, save_path: Path, n: int = 4) -> None:
    """Generate Grad-CAM overlays for sample images."""
    class_names = getattr(data_loader.dataset, "classes", ["NORMAL", "PNEUMONIA"])
    mean = config["data"]["normalize_mean"]
    std = config["data"]["normalize_std"]

    model.eval()
    images, labels = next(iter(data_loader))
    images = images.to(device)
    labels = labels.numpy().astype(int)

    target_layer = get_target_layer(model_name, model)
    grad_cam = GradCAM(model, target_layer)

    n = min(n, len(images))
    plt.figure(figsize=(12, 3.5 * n))

    for i in range(n):
        input_tensor = images[i : i + 1]
        cam = grad_cam.generate(input_tensor)

        with torch.no_grad():
            prob = torch.sigmoid(model(input_tensor)).item()
        pred = int(prob >= 0.5)

        img = denormalize_image(images[i].cpu(), mean, std)

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Original | True: {class_names[labels[i]]}")

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(img)
        plt.imshow(cam, cmap="jet", alpha=0.4)
        plt.axis("off")
        plt.title(f"Grad-CAM | Pred: {class_names[pred]} ({prob:.2f})")

    grad_cam.close()
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def save_metrics(metrics: Dict, path: Path) -> None:
    """Persist metrics dictionary to JSON file."""
    save_json(metrics, str(path))


def load_checkpoint_model(checkpoint_path: Path, default_model_name: str, device: torch.device):
    """Load model checkpoint and return initialized model ready for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", default_model_name)
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, model_name


def main():
    """Evaluate saved centralized/federated models and generate comparison artifacts."""
    from data_loader import get_centralized_loaders

    project_root = Path(__file__).resolve().parents[1]
    config = load_config(str(project_root / "configs" / "config.yaml"))
    device = get_device()

    _, _, test_loader = get_centralized_loaders(config, project_root)

    models_dir = project_root / config["paths"]["models_dir"]
    logs_dir = project_root / config["paths"]["logs_dir"]
    plots_dir = project_root / config["paths"]["plots_dir"]

    centralized_ckpt = models_dir / "centralized_best.pt"
    federated_ckpt = models_dir / "federated_best.pt"

    if not centralized_ckpt.exists() or not federated_ckpt.exists():
        raise FileNotFoundError(
            "Both centralized_best.pt and federated_best.pt are required. "
            "Run training scripts first."
        )

    central_model, central_model_name = load_checkpoint_model(
        centralized_ckpt,
        default_model_name=config["model"]["name"],
        device=device,
    )
    fed_model, fed_model_name = load_checkpoint_model(
        federated_ckpt,
        default_model_name=config["model"]["name"],
        device=device,
    )

    central_metrics = evaluate_model(central_model, test_loader, device)
    fed_metrics = evaluate_model(fed_model, test_loader, device)

    save_metrics(central_metrics, logs_dir / "centralized_metrics_eval.json")
    save_metrics(fed_metrics, logs_dir / "federated_metrics_eval.json")

    class_names = getattr(test_loader.dataset, "classes", ["NORMAL", "PNEUMONIA"])

    plot_confusion_matrix(
        central_metrics["confusion_matrix"],
        class_names,
        plots_dir / "centralized_confusion_matrix_eval.png",
        title="Centralized Model Confusion Matrix",
    )
    plot_confusion_matrix(
        fed_metrics["confusion_matrix"],
        class_names,
        plots_dir / "federated_confusion_matrix_eval.png",
        title="Federated Model Confusion Matrix",
    )

    y_true_c, y_prob_c = collect_predictions(central_model, test_loader, device)
    y_true_f, y_prob_f = collect_predictions(fed_model, test_loader, device)

    plot_roc_curve(
        y_true_c,
        y_prob_c,
        plots_dir / "centralized_roc_curve_eval.png",
        title="Centralized ROC Curve",
    )
    plot_roc_curve(
        y_true_f,
        y_prob_f,
        plots_dir / "federated_roc_curve_eval.png",
        title="Federated ROC Curve",
    )

    plot_metrics_bar(central_metrics, fed_metrics, plots_dir / "centralized_vs_federated_bar_eval.png")

    centralized_history_path = logs_dir / "centralized_history.json"
    federated_history_path = logs_dir / "federated_history.json"
    if centralized_history_path.exists() and federated_history_path.exists():
        with open(centralized_history_path, "r", encoding="utf-8") as f:
            centralized_history = json.load(f)
        with open(federated_history_path, "r", encoding="utf-8") as f:
            federated_history = json.load(f)
        plot_training_comparison(centralized_history, federated_history, plots_dir)

    plot_example_predictions(
        central_model,
        test_loader,
        device,
        config,
        plots_dir / "centralized_example_predictions_eval.png",
        n=8,
    )
    plot_example_predictions(
        fed_model,
        test_loader,
        device,
        config,
        plots_dir / "federated_example_predictions_eval.png",
        n=8,
    )

    plot_gradcam_examples(
        central_model,
        central_model_name,
        test_loader,
        device,
        config,
        plots_dir / "centralized_gradcam_examples_eval.png",
        n=4,
    )
    plot_gradcam_examples(
        fed_model,
        fed_model_name,
        test_loader,
        device,
        config,
        plots_dir / "federated_gradcam_examples_eval.png",
        n=4,
    )

    print("Evaluation complete. Metrics and plots were saved to outputs.")


if __name__ == "__main__":
    import json

    main()
