# 胸部 X 光肺炎分类（集中式学习 + 联邦学习）

[English Version](README.md)

这是一个基于 PyTorch 的可运行项目，用于胸部 X 光二分类（NORMAL / PNEUMONIA），包含：

- 集中式深度学习训练
- 联邦学习（FedAvg）与非独立同分布（Non-IID）客户端模拟
- 完整评估指标（Accuracy、Precision、Recall、F1、Confusion Matrix）
- 可视化结果（损失曲线、准确率曲线、混淆矩阵、ROC-AUC、对比柱状图）
- 可解释性扩展（Grad-CAM、样例预测可视化）

## 1) 项目结构

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

## 2) 环境配置（Anaconda）

请使用以下命令：

```bash
conda create -n fl_xray python=3.10
conda activate fl_xray
```

安装依赖：

```bash
pip install -r requirements.txt
```

或直接安装：

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas tqdm pyyaml numpy
```

## 3) 数据集目录格式

请将 Chest X-ray 数据集整理为 ImageFolder 格式：

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

## 4) 运行命令

集中式训练：

```bash
python src/train_centralized.py
```

联邦学习训练（FedAvg + Non-IID）：

```bash
python src/train_federated.py
```

可选：单独评估并重绘图表：

```bash
python src/evaluate.py
```

## 5) 训练流程说明

### 集中式训练

- 加载 train / val / test 全量数据
- 训练可选模型（simple_cnn 或 resnet18）
- 使用 BCEWithLogitsLoss 进行二分类优化
- 每个 epoch 记录训练和验证的 loss、accuracy
- 按验证集 loss 保存最佳模型

### 联邦学习训练（FedAvg）

- 将训练集拆分为 N 个客户端
- 使用类别比例偏斜模拟 Non-IID 分布
- 每轮下发全局模型给客户端进行本地训练
- 服务器按样本量加权聚合参数（FedAvg）
- 重复多轮后保存最佳全局模型

## 6) 输出结果

### 模型文件

- outputs/models/centralized_best.pt
- outputs/models/federated_best.pt

### 日志文件

- outputs/logs/centralized_history.json
- outputs/logs/federated_history.json
- outputs/logs/centralized_metrics.json
- outputs/logs/federated_metrics.json
- outputs/logs/comparison_metrics.json（若联邦训练时检测到集中式模型）

### 图像文件

- outputs/plots/loss_curve_comparison.png
- outputs/plots/accuracy_curve_comparison.png
- centralized / federated 混淆矩阵
- centralized / federated ROC 曲线与 AUC
- outputs/plots/centralized_vs_federated_bar.png
- Grad-CAM 可视化示例
- 样例预测图

## 7) 配置文件说明

请在 configs/config.yaml 中调整：

- 模型类型（simple_cnn 或 resnet18）
- batch size、学习率、训练轮数
- 联邦客户端数量、通信轮数、本地 epoch
- 图像归一化参数、数据加载并行参数

## 8) 结果解读建议

- 集中式训练通常收敛更快，在小数据场景下可能更高精度。
- 联邦学习在轮数充足时可逐步逼近集中式表现。
- Non-IID 会显著增加联邦优化难度，这也更贴近真实医疗场景。
- 建议结合混淆矩阵与 ROC-AUC 分析类别级表现，而不只看 accuracy。

## 9) 可复现性

- 项目已统一设置随机种子（Python / NumPy / PyTorch）
- 已开启 cuDNN 确定性配置
- 在相同配置下可减少实验波动
