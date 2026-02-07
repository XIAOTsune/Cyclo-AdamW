# Cyclo-AdamW 算法设计 (v1.1)

## 1. 核心物理原理：深度学习与经典力学的映射
**Cyclo-AdamW** 并非简单地模拟最速降线，而是将优化过程映射为在一个高维重力势能场中的质点运动。

### 1.1 势能 ($U$) <--> 损失 ($L$)
- **物理定义**：势能 $U$ 代表系统在当前位置 $x$ 具有的能量储备，即“距离全局极小值的高度”。
- **深度学习角色**：$L(x)$ (Loss) 直接映射为 $U(x)$ (Height)。
- **作用机制**：
    - **高势能 ($L \gg 0$)**：系统需要转化为高动能以快速“滑落”陡峭的斜坡。
    - **低势能 ($L \to 0$)**：系统动能自然衰减，防止过冲，类似于钟摆在最低点的微小震荡。
- **归一化问题**：由于不同任务 Loss 绝对值差异极大（MSE vs CE），我们引入“相对势能”，以 $L_{norm} = L / L_{0}$ 为准。

### 1.2 动能 ($K$) <--> 学习步长 ($\eta_{eff}$)
- **能量守恒**：$K + U = E_{total}$。假设初始 $v_0=0$，则 $v = \sqrt{2g(h_0 - h)}$。
    - 在这种模型下，Loss 越低，速度应该越快吗？**并非如此**。
    - **阻尼修正**：在优化中，我们不希望粒子真的像无阻尼钟摆一样永远震荡。我们引入**强阻尼**假设：速度并不完全由能量守恒决定，而是受限于当前的“势能储备”。
- **修正公式**：$v \propto \sqrt{L}$。
    - 这意味着：当前位置越高（离目标越远），允许的最大探索速度越大。
    - 这可以防止在平坦区域（高 Loss 但梯度小）陷入停滞，赋予参数“滑行”的能力。

## 2. 详细更新规则

### 2.1 摆线因子 (Cycloid Factor)
我们定义一个自适应因子 $\Phi(t)$ 来调节基础学习率。

$$ \Phi(t) = \sqrt{\frac{\bar{L}_{t}}{L_{initial} + \epsilon}} $$

- **$\bar{L}_{t}$ (EMA of Loss)**: 使用指数移动平均平滑 batch 间的随机性，代表“当前宏观高度”。
- **$L_{initial}$**: 训练开始时的 Loss，作为参考基准。
- **$\epsilon$**: 数值稳定项。

**具体作用**：
- **前期 ($\Phi \approx 1$)**：保持较高的有效学习率，利用高势能快速下降。
- **后期 ($\Phi \to 0$)**：随着 Loss 下降，$\Phi$ 自动衰减，提供一种基于任务进度的自然 Schedule，无需手动设置 `StepLR` 或 `CosineAnnealing`。
- **异常处理**：如果 $\Phi > 1$ (Loss 发散)，则强制 Clipping $\Phi \leftarrow 1$，防止不稳定。

### 2.2 量子阈值 ($h_{DL}$)
为了模拟量子隧穿效应并过滤热噪声，我们引入最小作用量原则。

$$ \mathcal{A} = \eta \cdot \|\mathbf{m}_t\| \cdot \|\nabla L_t\| $$

- **$h_{DL}$ 定义**：最小有效作用量 (Minimum Effective Action)。
- **物理意义**：普朗克常数。如果某次更新所做的“功”（步长 $\times$ 力）小于 $h_{DL}$，则该更新被视为“无效涨落”。

**过滤逻辑**：
如果不满足 $\mathcal{A} > h_{DL}$：
1. **硬阈值 (Hard Gating)**: 直接跳过 update ($\Delta \theta = 0$)。这会导致稀疏更新，可能加速推理但损害训练。
2. **软阈值 (Soft Gating - DEFAULT)**:
   $$ \text{Scale} = \text{sigmoid}(k \cdot (\mathcal{A} - h_{DL})) $$
   当作用量微小时，极度抑制步长，防止在极小值附近进行无意义的“布朗运动”式游走。

## 3. 算法伪代码 (Revised)

```python
# 初始化
loss_ema = initial_loss
params_t = params_0

For t in 1..T:
    g_t = grad(params_t)
    l_t = loss(params_t)
    
    # 1. Update Potential (Loss EMA)
    loss_ema = alpha * loss_ema + (1 - alpha) * l_t
    
    # 2. Calculate Cycloid Scale
    phi = sqrt(loss_ema / initial_loss)
    phi = clamp(phi, 0.1, 1.2) # 防止极端缩放
    
    # 3. AdamW Moment Updates
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    
    # 4. Action Check (Quantum Threshold)
    # Estimate effective step size without updates
    naive_step = lr * phi * m_t / (sqrt(v_t) + epsilon)
    action = norm(naive_step * g_t) # Dot product approx
    
    if action < h_DL:
        # Noise detected -> Suppress
        suppression_factor = action / (h_DL + epsilon) # Linear decay
        final_step = naive_step * suppression_factor
    else:
        final_step = naive_step

    # 5. Apply
    params_t = params_t - final_step - lr * weight_decay * params_t
```

## 4. 补充讨论

### 4.1 归一化与尺度不变性
- 不同的 Loss Scale 会显著影响 $\Phi$。
- **解决方案**: 在第一个 Batch 自动校准 $L_{initial}$。对后续的 Loss 使用相对变化量。

### 4.2 热启动 (Warmup)
- 在最开始的几步，Loss 可能剧烈波动。
- **策略**: 前 $N$ 步 (e.g., 1000) 禁用 Cycloid Factor 和 $h_{DL}$，退化为标准 AdamW，让 $L_{initial}$ 和 EMA 稳定下来。
