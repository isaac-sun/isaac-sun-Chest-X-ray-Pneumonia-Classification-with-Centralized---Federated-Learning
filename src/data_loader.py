from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms


def _compute_client_class1_quotas(
    client_sizes: np.ndarray,
    total_class1: int,
    min_ratio: float,
    max_ratio: float,
    min_samples_per_class: int,
    shuffle_target_ratios: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute per-client class-1 quotas with bounded skew and exact global conservation."""
    num_clients = len(client_sizes)
    min_ratio = float(np.clip(min_ratio, 0.0, 1.0))
    max_ratio = float(np.clip(max_ratio, min_ratio, 1.0))

    # Start from smooth target ratios, then project to integer quotas.
    target_ratios = np.linspace(min_ratio, max_ratio, num_clients)
    if shuffle_target_ratios and rng is not None:
        rng.shuffle(target_ratios)
    target_c1 = np.rint(client_sizes * target_ratios).astype(np.int64)

    # Try to guarantee both classes per client when globally feasible.
    min_per_class = int(max(0, min_samples_per_class))
    feasible_two_class = (
        (client_sizes.min() >= 2 * min_per_class)
        and (total_class1 >= num_clients * min_per_class)
        and ((client_sizes.sum() - total_class1) >= num_clients * min_per_class)
    )
    if not feasible_two_class:
        min_per_class = 0

    lower = np.full(num_clients, min_per_class, dtype=np.int64)
    upper = client_sizes - min_per_class
    target_c1 = np.clip(target_c1, lower, upper)

    current_sum = int(target_c1.sum())
    diff = current_sum - int(total_class1)

    # Decrease quotas if we allocated too many class-1 samples.
    while diff > 0:
        candidates = np.where(target_c1 > lower)[0]
        if len(candidates) == 0:
            break
        # Reduce from clients currently having highest class-1 share.
        j = candidates[np.argmax(target_c1[candidates] / np.maximum(client_sizes[candidates], 1))]
        target_c1[j] -= 1
        diff -= 1

    # Increase quotas if we allocated too few class-1 samples.
    while diff < 0:
        candidates = np.where(target_c1 < upper)[0]
        if len(candidates) == 0:
            break
        # Increase clients currently having lowest class-1 share.
        j = candidates[np.argmin(target_c1[candidates] / np.maximum(client_sizes[candidates], 1))]
        target_c1[j] += 1
        diff += 1

    return target_c1.astype(np.int64)


def build_transforms(config: Dict, train: bool = False):
    """Build image transforms for train/eval splits."""
    image_size = config["data"]["image_size"]
    mean = config["data"]["normalize_mean"]
    std = config["data"]["normalize_std"]

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_datasets(config: Dict, project_root: Path):
    """Create ImageFolder datasets for train/val/test."""
    data_root = project_root / config["data"]["root_dir"]

    train_dir = data_root / config["data"]["train_dir"]
    val_dir = data_root / config["data"]["val_dir"]
    test_dir = data_root / config["data"]["test_dir"]

    train_dataset = datasets.ImageFolder(train_dir, transform=build_transforms(config, train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_transforms(config, train=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=build_transforms(config, train=False))

    return train_dataset, val_dataset, test_dataset


def get_centralized_loaders(config: Dict, project_root: Path):
    """Create centralized dataloaders for all splits."""
    train_dataset, val_dataset, test_dataset = get_datasets(config, project_root)
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    # pin_memory only benefits CUDA and triggers warnings on MPS/CPU.
    pin_memory = bool(config["data"]["pin_memory"]) and torch.cuda.is_available()

    use_weighted_sampler = bool(config.get("training", {}).get("use_weighted_sampler", True))
    sampler = None
    train_shuffle = True
    if use_weighted_sampler:
        targets = np.array(train_dataset.targets)
        class_counts = np.bincount(targets, minlength=2).astype(np.float64)
        class_counts[class_counts == 0] = 1.0
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

        print(
            f"[Centralized Sampler] enabled=True, class0_count={int(class_counts[0])}, "
            f"class1_count={int(class_counts[1])}, class0_weight={class_weights[0]:.6f}, "
            f"class1_weight={class_weights[1]:.6f}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def split_noniid_indices(
    train_dataset,
    num_clients: int,
    seed: int = 42,
    ratio_span: float = 0.10,
    min_ratio_floor: float = 0.20,
    max_ratio_cap: float = 0.85,
    min_samples_per_class: int = 1,
    shuffle_target_ratios: bool = True,
) -> List[List[int]]:
    """Simulate non-IID binary label skew by giving each client a different class ratio."""
    rng = np.random.default_rng(seed)
    targets = np.array(train_dataset.targets)

    class0_idx = np.where(targets == 0)[0]
    class1_idx = np.where(targets == 1)[0]
    rng.shuffle(class0_idx)
    rng.shuffle(class1_idx)

    num_samples = len(targets)
    base_size = num_samples // num_clients
    remainder = num_samples % num_clients
    client_sizes = np.array([base_size + (1 if i < remainder else 0) for i in range(num_clients)], dtype=np.int64)

    # Use a feasible ratio band centered around global class-1 ratio.
    global_ratio = len(class1_idx) / max(num_samples, 1)
    ratio_span = float(max(0.0, ratio_span))
    min_ratio_floor = float(np.clip(min_ratio_floor, 0.0, 1.0))
    max_ratio_cap = float(np.clip(max_ratio_cap, 0.0, 1.0))
    min_ratio = max(min_ratio_floor, global_ratio - ratio_span)
    max_ratio = min(max_ratio_cap, global_ratio + ratio_span)
    if max_ratio < min_ratio:
        midpoint = float(np.clip(global_ratio, 0.0, 1.0))
        min_ratio = midpoint
        max_ratio = midpoint

    min_samples_per_class = int(max(0, min_samples_per_class))

    target_c1 = _compute_client_class1_quotas(
        client_sizes=client_sizes,
        total_class1=len(class1_idx),
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        min_samples_per_class=min_samples_per_class,
        shuffle_target_ratios=shuffle_target_ratios,
        rng=rng,
    )
    target_c0 = client_sizes - target_c1

    ptr0, ptr1 = 0, 0
    client_indices = []
    for client_id in range(num_clients):
        c0_count = int(target_c0[client_id])
        c1_count = int(target_c1[client_id])
        c0_slice = class0_idx[ptr0 : ptr0 + c0_count]
        c1_slice = class1_idx[ptr1 : ptr1 + c1_count]
        ptr0 += c0_count
        ptr1 += c1_count

        allocated = np.concatenate([c0_slice, c1_slice]).astype(np.int64)
        rng.shuffle(allocated)
        client_indices.append(allocated.tolist())

    return client_indices


def get_federated_client_loaders(config: Dict, project_root: Path):
    """Create local train loaders for each simulated federated client."""
    train_dataset, val_dataset, test_dataset = get_datasets(config, project_root)

    num_clients = config["federated"]["num_clients"]
    seed = config["seed"]
    num_workers = config["data"]["num_workers"]
    # pin_memory only benefits CUDA and triggers warnings on MPS/CPU.
    pin_memory = bool(config["data"]["pin_memory"]) and torch.cuda.is_available()
    client_batch_size = config["federated"]["client_batch_size"]

    client_indices = split_noniid_indices(
        train_dataset,
        num_clients=num_clients,
        seed=seed,
        ratio_span=config["federated"].get("noniid_ratio_span", 0.10),
        min_ratio_floor=config["federated"].get("noniid_min_ratio_floor", 0.20),
        max_ratio_cap=config["federated"].get("noniid_max_ratio_cap", 0.85),
        min_samples_per_class=config["federated"].get("noniid_min_samples_per_class", 1),
        shuffle_target_ratios=config["federated"].get("noniid_shuffle_target_ratios", True),
    )

    # Print client class ratios for quick sanity checks during FL runs.
    targets = np.array(train_dataset.targets)
    for client_id, idx_list in enumerate(client_indices):
        client_targets = targets[np.array(idx_list, dtype=np.int64)]
        class1_ratio = float((client_targets == 1).mean()) if len(client_targets) > 0 else 0.0
        print(
            f"[FL Split] Client {client_id}: samples={len(idx_list)}, "
            f"class0={(client_targets == 0).sum()}, class1={(client_targets == 1).sum()}, "
            f"class1_ratio={class1_ratio:.4f}"
        )

    client_loaders = []
    for idx_list in client_indices:
        subset = Subset(train_dataset, idx_list)
        loader = DataLoader(
            subset,
            batch_size=client_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        client_loaders.append(loader)

    eval_batch_size = config["training"]["batch_size"]
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return client_loaders, val_loader, test_loader
