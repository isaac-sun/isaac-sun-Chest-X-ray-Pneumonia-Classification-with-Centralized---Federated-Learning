# Modification Log

## 2026-03-27

### 1) Federated split strategy hardened to avoid class-collapse clients
- File: `src/data_loader.py`
- Changes:
  - Added `_compute_client_class1_quotas(...)` to generate per-client class-1 sample quotas with global count conservation.
  - Enforced per-client minimum samples from each class when feasible.
  - Replaced previous ad-hoc leftover filling logic that could create single-class clients.
  - Updated non-IID ratio generation to use a feasible band centered around global class ratio.
  - Added client split summary printouts in `get_federated_client_loaders(...)`.
- Why:
  - Previous split created extreme clients (including pure positive-class clients), which caused federated model collapse.

### 2) Added class-imbalance weighting for centralized training
- File: `src/train_centralized.py`
- Changes:
  - Added `compute_pos_weight_from_targets(...)`.
  - Switched loss to `BCEWithLogitsLoss(pos_weight=...)` computed from training-set class counts.
- Why:
  - Reduce majority-class bias and improve minority-class decision boundary.

### 3) Added class-imbalance weighting for federated local training
- Files: `src/client.py`, `src/train_federated.py`
- Changes:
  - `FederatedClient.train(...)` now accepts `pos_weight` and uses weighted BCE.
  - Added `compute_pos_weight_from_client_loaders(...)` in federated orchestrator.
  - Federated main loop now computes one global `pos_weight` and passes it to all clients.
- Why:
  - Mitigate class imbalance amplification during local updates in FedAvg.

### 4) Verification executed
- Syntax check: `python -m py_compile src/*.py` passed.
- Post-change split distribution check (5 clients):
  - client_0: p1=0.5431
  - client_1: p1=0.6433
  - client_2: p1=0.7430
  - client_3: p1=0.8428
  - client_4: p1=0.9425
- Result:
  - No single-class client remains.

### 5) Centralized training hardened for re-run
- Files: `src/data_loader.py`, `src/train_centralized.py`
- Changes:
  - Added optional `WeightedRandomSampler` in centralized loader (enabled by default) to mitigate class imbalance during mini-batch sampling.
  - Added sampler diagnostic prints (class counts and sample weights).
  - Added `ReduceLROnPlateau` scheduler in centralized training and logged current learning rate each epoch.
- Why:
  - Centralized run had unstable validation behavior under imbalanced data.
  - Weighted sampling + adaptive LR improve optimization stability for the re-run.

### 6) Verification executed (centralized updates)
- Syntax check: `python -m py_compile src/*.py` passed.
- Data split sanity check still passed after updates.

### 7) Data loader bugfix hardening (requested)
- Files: `src/data_loader.py`, `configs/config.yaml`
- Changes:
  - Added configurable non-IID controls in config:
    - `noniid_ratio_span`
    - `noniid_min_ratio_floor`
    - `noniid_max_ratio_cap`
    - `noniid_min_samples_per_class`
    - `noniid_shuffle_target_ratios`
  - Updated `split_noniid_indices(...)` to read these controls and apply bounds safely.
  - Added ratio-target shuffling support inside quota generation to avoid fixed extreme client IDs.
  - Kept exact sample conservation across all clients.
- Why:
  - Previous split still produced near-single-class clients in heavily imbalanced data.
  - Needed tighter/controlled skew and better client-level diversity for stable FedAvg.

---

## How to keep recording future edits
- Append a new dated section (`## YYYY-MM-DD`) for every change batch.
- For each entry, include:
  - File(s)
  - What changed
  - Why changed
  - How verified
