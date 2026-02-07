# Cyclo-AdamW 算法设计 (v2.1)

## 1. 核心物理原理：深度学习与经典力学的映射
**Cyclo-AdamW** 并非简单地模拟最速降线，而是将优化过程映射为在一个高维重力势能场中的质点运动。

### 1.1 势能 ($U$) <--> 损失 ($L$)
- **物理定义**：势能 $U$ 代表系统在当前位置 $x$ 具有的能量储备，即“距离全局极小值的高度”。
- **深度学习角色**：$L(x)$ (Loss) 直接映射为 $U(x)$ (Height)。
- **作用机制**：
    - **高势能 ($L \gg 0$)**：系统需要转化为高动能以快速“滑落”陡峭的斜坡。
    - **极低势能 ($L \to 0$)**：系统动能自然衰减，防止过冲，类似于钟摆在最低点的微小震荡。
- **动态校准 (Auto-Calibration)** (V3): 如果训练过程中 Loss 突然升高（如进入新的高地），算法会自动重置 $L_{initial} = L_{current}$，防止势能计算溢出并重新激活下降动能。

### 1.2 动能与摆线因子 ($\Phi$)
传统的动能公式 $v \propto \sqrt{L}$ 在深度学习中表现过于激进。我们引入 **能量保留因子 (Gamma, $\gamma$)** 来调节势能转化为动能的效率。

$$ \Phi(t) = \left( \frac{\bar{L}_{t}}{L_{initial} + \epsilon} \right)^\gamma $$

- **$\gamma$ (Gamma)**: 能量保留系数。
    - $\gamma = 0.5$ (物理上的自由落体): 衰减极快，适合凸优化。
    - $\gamma = 0.25$ (默认，工程优化): 保留更多动能，防止在训练中期过早减速 (如 CIFAR-10 任务中观察到的)，从而匹配 AdamW 的收敛速度。

## 2. 详细更新规则

### 2.1 摆线因子 (Cycloid Factor)
我们定义一个自适应因子 $\Phi(t)$ 来调节基础学习率。

- **$\bar{L}_{t}$ (EMA of Loss)**: 使用指数移动平均平滑 batch 间的随机性。
- **具体作用**：
    - **前期 ($\Phi \approx 1$)**：保持较高的有效学习率。
    - **后期 ($\Phi \to 0$)**：随着 Loss 下降，$\Phi$ 自动衰减，提供一种基于任务进度的自然 Schedule。

### 2.2 量子阈值 ($h_{DL}$) 与平均作用量密度 (Mean Action Density) (V2)
为了模拟量子隧穿效应并过滤热噪声，我们引入最小作用量原则。相比 V1 的总作用量，V2 引入了 **平均作用量密度** 以实现尺度不变性 (Scale Invariance)。

$$ \mathcal{A}_{density} = \frac{1}{N} \sum^{N}_{i=1} |\text{step}_i \cdot \text{grad}_i| $$

其中 $\text{step}_i$ 是未经过滤的建议步长。

- **$h_{DL}$ 定义**：最小有效作用量密度。
- **物理意义**：普朗克常数。如果某次更新中，每个参数平均所做的“功”小于 $h_{DL}$，则该更新被视为“无效涨落”。
- **V2 改进**: 使用密度 ($1/N$) 而非总和，使得该机制对于不同大小的层（如 64通道卷积层 vs 偏置向量）都具有相同的物理意义，避免了小参数层被误杀。

**过滤逻辑 (Soft Gating)**：
如果不满足 $\mathcal{A}_{density} > h_{DL}$：
   $$ \text{Scale} = \frac{\mathcal{A}_{density}}{h_{DL} + \epsilon} $$
   这将线性抑制微小的噪声更新，稳定训练。

## 3. 算法伪代码 (Revised V2.1)

```python
# 初始化
loss_ema = initial_loss
params_t = params_0

For t in 1..T:
    g_t = grad(params_t)
    l_t = loss(params_t)
    
    # 1. Update Potential (Loss EMA) & Auto-Calibration
    loss_ema = alpha * loss_ema + (1 - alpha) * l_t
    if loss_ema > initial_loss: initial_loss = loss_ema # (V3)
    
    # 2. Calculate Cycloid Scale with Gamma
    ratio = loss_ema / initial_loss
    phi = ratio ^ gamma  # (V3: Gamma tuning)
    phi = clamp(phi, 0.1, 1.2)
    
    # 3. AdamW Moment Updates (With Bias Correction - V2.1 Fix)
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    
    # 4. Action Check (Quantum Threshold V2)
    # Estimate effective step size
    step_size = lr * phi * sqrt(1 - beta2^t) / (1 - beta1^t)
    work_density = (m_hat / (sqrt(v_hat) + eps)) * g_t
    
    # Mean Action Density (Scale Invariant)
    mean_action = step_size * mean(|work_density|)
    
    if mean_action < h_DL:
        # Noise detected -> Suppress
        suppression_factor = mean_action / (h_DL + epsilon)
        final_step_size = step_size * suppression_factor
    else:
        final_step_size = step_size

    # 5. Apply
    params_t = params_t - final_step_size * (m_hat / (sqrt(v_hat) + eps)) - lr * wa * params_t
```
