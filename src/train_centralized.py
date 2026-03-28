from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data_loader import get_centralized_loaders
from evaluate import (
    collect_predictions,
    evaluate_model,
    plot_confusion_matrix,
    plot_example_predictions,
    plot_gradcam_examples,
    plot_roc_curve,
    save_metrics,
)
from model import build_model
from utils import ensure_dir, epoch_binary_accuracy, get_device, load_config, set_seed


def compute_pos_weight_from_targets(targets, device: torch.device) -> torch.Tensor:
    """Compute BCE pos_weight = num_negative / num_positive from dataset targets."""
    labels = torch.tensor(targets, dtype=torch.float32)
    num_pos = float((labels == 1).sum().item())
    num_neg = float((labels == 0).sum().item())
    if num_pos <= 0:
        return torch.tensor(1.0, device=device)
    return torch.tensor(num_neg / num_pos, dtype=torch.float32, device=device)


def get_dataset_targets(dataset):
    """Extract integer labels from dataset with a safe fallback path."""
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return targets
    return [int(dataset[i][1]) for i in range(len(dataset))]


def train_one_epoch(model, data_loader, optimizer, criterion, device: torch.device, epoch_idx: int, total_epochs: int):
    """Run one centralized training epoch and return mean loss/accuracy."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batches = 0

    progress = tqdm(
        data_loader,
        desc=f"Train Epoch {epoch_idx + 1}/{total_epochs}",
        leave=True,
        dynamic_ncols=True,
        ascii=True,
    )
    for images, labels in progress:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        logits = model(images).view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_acc = epoch_binary_accuracy(logits.detach(), labels)
        running_loss += batch_loss
        running_acc += batch_acc
        batches += 1
        progress.set_postfix(loss=f"{batch_loss:.4f}", acc=f"{batch_acc:.4f}")

    return running_loss / max(1, batches), running_acc / max(1, batches)


@torch.no_grad()
def evaluate_epoch(model, data_loader, criterion, device: torch.device):
    """Evaluate model for one epoch and return mean loss/accuracy."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    batches = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.float().to(device)

        logits = model(images).view(-1)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        running_acc += epoch_binary_accuracy(logits, labels)
        batches += 1

    return running_loss / max(1, batches), running_acc / max(1, batches)


def main():
    """Train a centralized binary classifier and persist checkpoints/plots/metrics."""
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(str(project_root / "configs" / "config.yaml"))

    set_seed(config["seed"])
    device = get_device()

    for key in ["models_dir", "plots_dir", "logs_dir"]:
        ensure_dir(str(project_root / config["paths"][key]))

    train_loader, val_loader, test_loader = get_centralized_loaders(config, project_root)

    model = build_model(config["model"]["name"], pretrained=config["model"]["pretrained"]).to(device)
    train_targets = get_dataset_targets(train_loader.dataset)
    pos_weight = compute_pos_weight_from_targets(train_targets, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_state = None

    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(float(train_loss))
        history["train_accuracy"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    models_dir = project_root / config["paths"]["models_dir"]
    logs_dir = project_root / config["paths"]["logs_dir"]
    plots_dir = project_root / config["paths"]["plots_dir"]

    best_model_path = models_dir / "centralized_best.pt"
    torch.save(
        {
            "model_name": config["model"]["name"],
            "state_dict": model.state_dict(),
        },
        best_model_path,
    )
    print(f"Saved best centralized model to: {best_model_path}")

    with open(logs_dir / "centralized_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    test_metrics = evaluate_model(model, test_loader, device)
    save_metrics(test_metrics, logs_dir / "centralized_metrics.json")

    y_true, y_prob = collect_predictions(model, test_loader, device)
    class_names = getattr(test_loader.dataset, "classes", ["NORMAL", "PNEUMONIA"])

    plot_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        plots_dir / "centralized_confusion_matrix.png",
        title="Centralized Model Confusion Matrix",
    )
    roc_auc = plot_roc_curve(
        y_true,
        y_prob,
        plots_dir / "centralized_roc_curve.png",
        title="Centralized Model ROC Curve",
    )
    print(f"Centralized ROC-AUC: {roc_auc:.4f}")

    plot_example_predictions(
        model,
        test_loader,
        device,
        config,
        plots_dir / "centralized_example_predictions.png",
        n=8,
    )
    plot_gradcam_examples(
        model,
        config["model"]["name"],
        test_loader,
        device,
        config,
        plots_dir / "centralized_gradcam_examples.png",
        n=4,
    )

    print("Centralized training and evaluation complete.")


if __name__ == "__main__":
    main()
