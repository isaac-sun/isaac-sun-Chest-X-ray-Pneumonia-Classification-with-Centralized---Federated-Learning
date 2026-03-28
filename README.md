# Chest X-ray Pneumonia Classification with Centralized + Federated Learning

[中文文档 (Chinese Version)](README_ZH.md)

Production-quality PyTorch project for binary pneumonia classification on chest X-ray images, with:

- Centralized deep learning training
- Federated learning (FedAvg) with non-IID client simulation
- Full evaluation (accuracy, precision, recall, F1, confusion matrix)
- Visualization suite (loss/accuracy curves, confusion matrix, ROC-AUC, bar comparison)
- Bonus explainability (Grad-CAM and example predictions)

## 1) Project Structure

```text
project_root/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train_centralized.py
│   ├── train_federated.py
│   ├── client.py
│   ├── server.py
│   ├── utils.py
│   └── evaluate.py
├── outputs/
│   ├── models/
│   ├── plots/
│   └── logs/
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## 2) Environment Setup (Anaconda)

Use exactly:

```bash
conda create -n fl_xray python=3.10
conda activate fl_xray
```

Install required packages (Conda + pip hybrid is fine):

```bash
pip install -r requirements.txt
```

Or directly:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas tqdm pyyaml numpy
```

## 3) Dataset Layout

Important: The dataset is not uploaded to this repository.
Please download the Chest X-ray dataset yourself and organize it as follows.

Place the Chest X-ray dataset in ImageFolder format:

```text
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

If your downloaded dataset has a different folder split or class naming, adjust it to match
the structure above before training.

## 4) Run Commands

Centralized training:

```bash
python src/train_centralized.py
```

Federated training (FedAvg with non-IID clients):

```bash
python src/train_federated.py
```

Optional standalone evaluation/regeneration of plots:

```bash
python src/evaluate.py
```

## 5) What The Pipeline Does

### Centralized

- Loads full train/val/test datasets
- Trains selected model (`simple_cnn` or `resnet18`)
- Uses `BCEWithLogitsLoss` for binary classification
- Tracks train/val loss and accuracy per epoch
- Saves best model by validation loss

### Federated (FedAvg)

- Splits train data into `N` non-IID clients with different class ratios
- Each client receives global model and trains locally for `E` epochs
- Server aggregates client weights with sample-size weighted FedAvg
- Repeats for configured rounds
- Saves best global model by validation loss

## 6) Outputs

### Models

- `outputs/models/centralized_best.pt`
- `outputs/models/federated_best.pt`

### Logs

- `outputs/logs/centralized_history.json`
- `outputs/logs/federated_history.json`
- `outputs/logs/centralized_metrics.json`
- `outputs/logs/federated_metrics.json`
- `outputs/logs/comparison_metrics.json` (if centralized exists when federated is run)

### Plots

- Loss curve comparison: `outputs/plots/loss_curve_comparison.png`
- Accuracy curve comparison: `outputs/plots/accuracy_curve_comparison.png`
- Confusion matrices (centralized + federated)
- ROC curves and AUC (centralized + federated)
- Comparison bar chart: `outputs/plots/centralized_vs_federated_bar.png`
- Grad-CAM examples
- Sample prediction panels

## 7) Config Notes

Edit `configs/config.yaml` to control:

- Model type (`simple_cnn` or `resnet18`)
- Batch size, learning rate, epochs
- Number of FL clients, rounds, local epochs
- Data normalization and worker settings

## 8) Interpreting Results

Typical behavior to expect:

- Centralized model often converges faster and may reach higher accuracy in low-data FL settings.
- Federated model can approach centralized performance with enough rounds and local epochs.
- Non-IID client splits generally make FL optimization harder; this is realistic for healthcare silos.
- ROC-AUC and confusion matrix reveal class-wise behavior beyond plain accuracy.

## 9) Reproducibility

- Global seed is applied for Python, NumPy, and PyTorch
- cuDNN deterministic settings are enabled
- Use same seed/config to reduce run-to-run variance
