from copy import deepcopy
from typing import Dict, List

import torch


def fedavg(client_updates: List[Dict]) -> Dict[str, torch.Tensor]:
    """Aggregate client model weights using sample-size weighted FedAvg."""
    if not client_updates:
        raise ValueError("No client updates provided for aggregation.")

    total_samples = sum(update["num_samples"] for update in client_updates)
    if total_samples <= 0:
        raise ValueError("Total number of samples across clients must be positive.")

    avg_state = deepcopy(client_updates[0]["state_dict"])
    for key in avg_state.keys():
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)

    for update in client_updates:
        weight = update["num_samples"] / total_samples
        local_state = update["state_dict"]
        for key in avg_state.keys():
            avg_state[key] += local_state[key].float() * weight

    return avg_state
