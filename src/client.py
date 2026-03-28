from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from model import build_model
from utils import epoch_binary_accuracy, state_dict_to_cpu


class FederatedClient:
    """Federated client that performs local model training on its private subset."""

    def __init__(self, client_id: int, train_loader, config: Dict, device: torch.device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.local_epochs = config["federated"]["local_epochs"]
        self.lr = config["federated"]["client_lr"]
        self.weight_decay = config["federated"]["client_weight_decay"]
        self.model_name = config["model"]["name"]
        self.pretrained = config["model"]["pretrained"]

    def train(self, global_state: Dict[str, torch.Tensor], round_idx: int, pos_weight: torch.Tensor):
        """Train local model initialized from global state and return updated weights."""
        model = build_model(self.model_name, pretrained=self.pretrained).to(self.device)
        model.load_state_dict(deepcopy(global_state))

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        epoch_losses = []
        epoch_accs = []

        for local_epoch in range(self.local_epochs):
            running_loss = 0.0
            running_acc = 0.0
            total_batches = 0

            progress = tqdm(
                self.train_loader,
                desc=f"Client {self.client_id} | Round {round_idx + 1} | Epoch {local_epoch + 1}",
                leave=True,
                dynamic_ncols=True,
                ascii=True,
            )
            for images, labels in progress:
                images = images.to(self.device)
                labels = labels.float().to(self.device)

                optimizer.zero_grad()
                logits = model(images).view(-1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                acc = epoch_binary_accuracy(logits.detach(), labels)
                running_loss += loss.item()
                running_acc += acc
                total_batches += 1
                progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

            epoch_losses.append(running_loss / max(total_batches, 1))
            epoch_accs.append(running_acc / max(total_batches, 1))

        return {
            "client_id": self.client_id,
            "num_samples": len(self.train_loader.dataset),
            "state_dict": state_dict_to_cpu(model.state_dict()),
            "loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
            "accuracy": float(epoch_accs[-1]) if epoch_accs else 0.0,
        }
