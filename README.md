# Cyclo-AdamW: Physics-Inspired Optimizer for Deep Learning

**Cyclo-AdamW** is a novel optimization algorithm that bridges **Classical Mechanics** (Brachistochrone problem) and **Deep Learning**. By modeling the loss landscape as a gravitational potential field, it dynamically adjusts the learning rate and filters noise, achieving faster convergence and better generalization.

![Cycloid Animation](https://upload.wikimedia.org/wikipedia/commons/3/37/Brachistochrone.gif)
*(Concept: The Cycloid curve is the fastest path under gravity)*

## 🚀 Key Features

1.  **Cycloid Factor (摆线因子)**:
    - Dynamically scales the step size based on the "Potential Energy" (Loss) relative to the initial state.
    - **High Loss** $\rightarrow$ High Potential $\rightarrow$ Faster Descent.
    - **Low Loss** $\rightarrow$ Low Potential $\rightarrow$ Automatic Decay.

2.  **Quantum Threshold (量子阈值 $h_{DL}$)**:
    - Inspired by Planck's constant.
    - Filters out "thermal noise" updates where the **Action** ($Step \times Gradient$) is below a minimum threshold ($h_{DL}$).
    - Stabilizes training in flat or noisy regions.

## 📊 Performance (Verified)

| Task | Metric | AdamW | Cyclo-AdamW | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Non-Convex Opt** (Rosenbrock) | Final Loss | 3.9495 | **3.3123** | **-16% Loss** |
| **Image Classif** (MNIST) | Accuracy | 98.77% | **99.00%** | **+0.23% Acc** |

> *See [Verification Report](docs/verification_report.md) for details.*

## 📦 Installation

Copy the `src/cyclo_adamw.py` file to your project, or clone this repository:

```bash
git clone https://github.com/your-username/Cyclo-AdamW.git
cd Cyclo-AdamW
pip install -r requirements.txt
```

## 🛠 Usage

It functions as a drop-in replacement for `torch.optim.AdamW`.

```python
from src.cyclo_adamw import CycloAdamW

# Initialize Optimizer
optimizer = CycloAdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    h_dl=1e-5,          # Quantum Threshold (Default: 1e-5)
    warmup_steps=100    # Warmup steps before physics logic activates
)

# Training Loop
def train(model, loader, optimizer):
    model.train()
    for data, target in loader:
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            return loss

        # Note: CycloAdamW requires a closure for loss re-evaluation if needed,
        # though current implementation is efficient.
        optimizer.step(closure)
```

## 📂 Project Structure

```
Cyclo-AdamW/
├── src/
│   └── cyclo_adamw.py    # The core optimizer implementation
├── tests/
│   ├── test_convex.py    # Verification on mathematical functions
│   └── test_mnist.py     # Verification on MNIST dataset
├── docs/
│   ├── algorithm_design.md  # Mathematical formulation
│   └── verification_report.md
└── requirements.txt
```

## 📝 Citation
If you use this optimizer in your research, please cite:

```bibtex
@misc{CycloAdamW2026,
  author = {Your Name},
  title = {Cyclo-AdamW: A Physics-Inspired Optimizer},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/Cyclo-AdamW}}
}
```

## 📄 License
This project is licensed under the **MIT License**.
