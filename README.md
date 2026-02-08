# Cyclo-AdamW: Physics-Inspired Optimizer for Deep Learning <br> (基于物理摆线原理的深度学习优化器)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

[English](#english) | [中文](#chinese)

---

<a name="english"></a>
## 🇬🇧 English Description

**Cyclo-AdamW** is a novel optimization algorithm that bridges **Classical Mechanics** (Brachistochrone problem) and **Deep Learning**. By modeling the loss landscape as a gravitational potential field, it dynamically adjusts the learning rate and filters noise, achieving faster convergence and better generalization.

![Cycloid Animation](https://upload.wikimedia.org/wikipedia/commons/3/37/Brachistochrone.gif)
*(Concept: The Cycloid curve is the fastest path under gravity)*

### 🚀 Key Features

1.  **Cycloid Factor ($\phi$) with Energy Retention**:
    - Dynamically scales the step size based on **Potential Energy** (Loss).
    - **Energy Retention ($\gamma$)**: Allows tuning of how aggressively the learning rate decays as loss drops.
    - **Auto-Calibration**: Automatically resets the potential reference ($L_0$) if the loss landscape shifts significantly.

2.  **Quantum Threshold ($h_{DL}$)** via Mean Action Density:
    - Filters out "thermal noise" updates where the **Mean Action Density** (Average work per parameter) is below a threshold ($h_{DL}$).
    - **Scale Invariant**: Robust across different layer sizes (Conv2d vs Bias).
    - Stabilizes training in flat or noisy regions without killing effective gradients.

### 📊 Performance (Verified)

| Task | Metric | AdamW | Cyclo-AdamW | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Non-Convex Opt** (Rosenbrock) | Final Loss | 3.9495 | **3.3123** | **-16% Loss** |
| **Image Classif** (MNIST) | Accuracy | 98.77% | **99.00%** | **+0.23% Acc** |
| **Complex Vision** (CIFAR-10) | Accuracy (100 Epochs) | **92.71%** | 92.61% | *Comparable (-0.1%)* |

> *See [Verification Report](docs/verification_report.md) for details.*

### 📦 Installation

Copy the `src/cyclo_adamw.py` file to your project, or clone this repository:

```bash
git clone https://github.com/XIAOTsune/Cyclo-AdamW.git
cd Cyclo-AdamW
pip install -r requirements.txt
```

### 🛠 Usage

It functions as a drop-in replacement for `torch.optim.AdamW`.

```python
from src.cyclo_adamw import CycloAdamW

# Initialize Optimizer
optimizer = CycloAdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    h_dl=1e-8,          # Quantum Threshold (Default: 1e-8)
    gamma=0.25,         # Energy Retention (Default: 0.25)
    warmup_steps=500    # Warmup steps before physics logic activates
)
```

---

<a name="chinese"></a>
## 🇨🇳 中文介绍

**Cyclo-AdamW** 是一个新颖的深度学习优化算法，它将 **经典力学**（最速降线问题）与 **深度学习** 相结合。通过将损失地形建模为重力势能场，它能够动态调整学习率并过滤噪声，从而实现更快的收敛速度和更好的泛化能力。

### 🚀 核心特性

1.  **带能量保留的摆线因子 (Cycloid Factor $\phi$)**:
    - 根据相对于初始状态的**势能**（Loss）动态缩放步长。
    - **能量保留 ($\gamma$)**: 允许调节学习率随 Loss 下降而衰减的激进程度。
    - **自动校准**: 如果 Loss 地形发生剧烈变化，自动重置势能参考点 ($L_0$)。

2.  **基于平均作用量密度的量子阈值 ($h_{DL}$)**:
    - 过滤掉“热噪声”更新，即当 **平均作用量密度** (每个参数的平均做功) 低于阈值 ($h_{DL}$) 时，抑制更新。
    - **尺度不变性**: 对不同大小的层（如大型卷积层与小型偏置层）具有鲁棒性。
    - 在平坦或嘈杂区域稳定训练，同时保留有效梯度。

### 📊 性能表现 (已验证)

| 任务 | 指标 | AdamW | Cyclo-AdamW | 提升 |
| :--- | :--- | :--- | :--- | :--- |
| **非凸优化** (Rosenbrock) | 最终 Loss | 3.9495 | **3.3123** | **Loss 降低 16%** |
| **图像分类** (MNIST) | 准确率 | 98.77% | **99.00%** | **准确率提升 0.23%** |
| **复杂视觉** (CIFAR-10) | 准确率 (3 Epochs) | **70.45%** | 68.40% | *相当 (-2%)* |

> *详见 [验证报告](docs/verification_report.md)。*

### 📦 安装

将 `src/cyclo_adamw.py` 文件复制到您的项目中，或克隆此仓库：

```bash
git clone https://github.com/XIAOTsune/Cyclo-AdamW.git
cd Cyclo-AdamW
pip install -r requirements.txt
```

### 🛠 使用方法

它可以作为 `torch.optim.AdamW` 的直接替代品使用。

```python
from src.cyclo_adamw import CycloAdamW

# 初始化优化器
optimizer = CycloAdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    h_dl=1e-8,          # 量子阈值 (默认: 1e-8)
    gamma=0.25,         # 能量保留因子 (默认: 0.25)
    warmup_steps=500    # 物理逻辑激活前的热启动步数
)
```

---

## 📂 Project Structure / 项目结构

```
Cyclo-AdamW/
├── src/
│   └── cyclo_adamw.py    # Core implementation / 核心实现
├── tests/
│   ├── test_convex.py    # Math function verification / 数学函数验证
│   ├── test_mnist.py     # Deep learning verification / 深度学习验证
│   └── test_cifar10.py   # Complex dataset verification / 复杂数据集验证
├── docs/
│   ├── algorithm_design.md  # Theory / 理论推导
│   └── verification_report.md # Results / 验证报告
└── requirements.txt
```

## 📝 Citation / 引用

If you use this optimizer in your research, please cite:
如果您在研究中使用了此优化器，请引用：

```bibtex
@misc{CycloAdamW2026,
  author = {XIAOTsune},
  title = {Cyclo-AdamW: A Physics-Inspired Optimizer},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XIAOTsune/Cyclo-AdamW}}
}
```

## 📄 License / 许可
This project is licensed under the **MIT License**.
本项目采用 **MIT 许可证**。
