import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from client import FederatedClient
from data_loader import get_federated_client_loaders
from evaluate import (
    collect_predictions,
    evaluate_model,
    plot_confusion_matrix,
    plot_example_predictions,
    plot_gradcam_examples,
    plot_metrics_bar,
    plot_roc_curve,
    plot_training_comparison,
    save_metrics,
)
from model import build_model
from server import fedavg
from utils import ensure_dir, epoch_binary_accuracy, get_device, load_config, set_seed
import numpy as np

def compute_pos_weight_from_client_loaders(client_loaders, device: torch.device) -> torch.Tensor:
    """Compute global pos_weight from the union of all client subsets."""
    num_pos = 0
    num_neg = 0
    for loader in client_loaders:
        subset = loader.dataset
        base_targets = np.array(subset.dataset.targets)
        subset_indices = np.array(subset.indices, dtype=np.int64)
        labels = base_targets[subset_indices]
        num_pos += int((labels == 1).sum())
        num_neg += int((labels == 0).sum())

    if num_pos <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device)
    return torch.tensor(num_neg / num_pos, dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate_global(model, data_loader, criterion, device: torch.device):
    """Evaluate global model on validation/test set."""
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


def maybe_load_centralized_model(project_root: Path, config, device: torch.device):
    """Load centralized checkpoint if available for comparison plots."""
    ckpt_path = project_root / config["paths"]["models_dir"] / "centralized_best.pt"
    if not ckpt_path.exists():
        return None

    checkpoint = torch.load(ckpt_path, map_location=device)
    model_name = checkpoint.get("model_name", config["model"]["name"])
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main():
    """Run Federated Averaging training with non-IID clients and save artifacts."""
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(str(project_root / "configs" / "config.yaml"))

    set_seed(config["seed"])
    device = get_device()

    for key in ["models_dir", "plots_dir", "logs_dir"]:
        ensure_dir(str(project_root / config["paths"][key]))

    client_loaders, val_loader, test_loader = get_federated_client_loaders(config, project_root)

    global_model = build_model(config["model"]["name"], pretrained=config["model"]["pretrained"]).to(device)
    pos_weight = compute_pos_weight_from_client_loaders(client_loaders, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    clients = [
        FederatedClient(client_id=i, train_loader=loader, config=config, device=device)
        for i, loader in enumerate(client_loaders)
    ]

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_global_state = None

    rounds = config["federated"]["rounds"]
    for round_idx in range(rounds):
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        client_updates = []
        progress = tqdm(clients, desc=f"Federated Round {round_idx + 1}/{rounds}")
        for client in progress:
            update = client.train(global_state, round_idx, pos_weight)
            client_updates.append(update)

        aggregated_state = fedavg(client_updates)
        global_model.load_state_dict(aggregated_state)

        avg_client_loss = sum(u["loss"] for u in client_updates) / max(1, len(client_updates))
        avg_client_acc = sum(u["accuracy"] for u in client_updates) / max(1, len(client_updates))

        val_loss, val_acc = evaluate_global(global_model, val_loader, criterion, device)

        history["train_loss"].append(float(avg_client_loss))
        history["train_accuracy"].append(float(avg_client_acc))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(val_acc))

        print(
            f"Round {round_idx + 1}/{rounds} | "
            f"Client Loss: {avg_client_loss:.4f}, Client Acc: {avg_client_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    if best_global_state is not None:
        global_model.load_state_dict(best_global_state)

    models_dir = project_root / config["paths"]["models_dir"]
    logs_dir = project_root / config["paths"]["logs_dir"]
    plots_dir = project_root / config["paths"]["plots_dir"]

    fed_model_path = models_dir / "federated_best.pt"
    torch.save(
        {
            "model_name": config["model"]["name"],
            "state_dict": global_model.state_dict(),
        },
        fed_model_path,
    )
    print(f"Saved best federated model to: {fed_model_path}")

    with open(logs_dir / "federated_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fed_test_metrics = evaluate_model(global_model, test_loader, device)
    save_metrics(fed_test_metrics, logs_dir / "federated_metrics.json")

    y_true_f, y_prob_f = collect_predictions(global_model, test_loader, device)
    class_names = getattr(test_loader.dataset, "classes", ["NORMAL", "PNEUMONIA"])

    plot_confusion_matrix(
        fed_test_metrics["confusion_matrix"],
        class_names,
        plots_dir / "federated_confusion_matrix.png",
        title="Federated Model Confusion Matrix",
    )
    roc_auc_fed = plot_roc_curve(
        y_true_f,
        y_prob_f,
        plots_dir / "federated_roc_curve.png",
        title="Federated Model ROC Curve",
    )
    print(f"Federated ROC-AUC: {roc_auc_fed:.4f}")

    plot_example_predictions(
        global_model,
        test_loader,
        device,
        config,
        plots_dir / "federated_example_predictions.png",
        n=8,
    )
    plot_gradcam_examples(
        global_model,
        config["model"]["name"],
        test_loader,
        device,
        config,
        plots_dir / "federated_gradcam_examples.png",
        n=4,
    )

    centralized_history_path = logs_dir / "centralized_history.json"
    centralized_model = maybe_load_centralized_model(project_root, config, device)

    if centralized_model is not None and centralized_history_path.exists():
        with open(centralized_history_path, "r", encoding="utf-8") as f:
            centralized_history = json.load(f)

        central_metrics = evaluate_model(centralized_model, test_loader, device)
        save_metrics(central_metrics, logs_dir / "centralized_metrics_from_fed_run.json")

        plot_training_comparison(centralized_history, history, plots_dir)
        plot_metrics_bar(central_metrics, fed_test_metrics, plots_dir / "centralized_vs_federated_bar.png")

        plot_confusion_matrix(
            central_metrics["confusion_matrix"],
            class_names,
            plots_dir / "centralized_confusion_matrix_from_fed_run.png",
            title="Centralized Model Confusion Matrix",
        )

        y_true_c, y_prob_c = collect_predictions(centralized_model, test_loader, device)
        plot_roc_curve(
            y_true_c,
            y_prob_c,
            plots_dir / "centralized_roc_curve_from_fed_run.png",
            title="Centralized Model ROC Curve",
        )

        comparison_payload = {
            "centralized": central_metrics,
            "federated": fed_test_metrics,
        }
        with open(logs_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
            json.dump(comparison_payload, f, indent=2)

        print("Generated centralized-vs-federated comparison artifacts.")
    else:
        print("Centralized model/history not found. Run centralized training first for full comparison plots.")

    print("Federated training and evaluation complete.")


if __name__ == "__main__":
    main()
