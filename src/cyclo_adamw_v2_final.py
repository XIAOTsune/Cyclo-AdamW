import torch
import math
from torch.optim.optimizer import Optimizer

class CycloAdamW(Optimizer):
    r"""Implements Cyclo-AdamW algorithm (V2 - Classic Physics).

    Cyclo-AdamW 算法实现 (V2 - 经典物理版)

    核心原理 (Core Principles):
    该算法将神经网络的优化过程建模为质点在重力势能场中的运动 (Brachistochrone Problem / 最速降线问题)。
    
    1. 势能与损失 (Potential Energy & Loss):
       - 损失函数 Loss 被视为重力势能 U (Potential Energy)。
       - Loss 越大，处于“高处”，势能越大。
    
    2. 摆线因子 (Cycloid Factor / Brachistochrone Law):
       - phi = sqrt(L_ema / L_0)
       - 根据能量守恒，质点滑落的速度 v 与高度的平方根成正比 (v ~ sqrt(2gh))。
       - 因此，步长 (Step Size) 动态调整为 phi 倍，模拟重力加速效果。
       
    3. 量子阈值 (Quantum Threshold / h_dl):
       - 引入最小作用量原理 (Principle of Least Action)。
       - 如果某次参数更新的“平均作用量密度”小于阈值 h_dl (类似普朗克常数)，则视为无效的热涨落 (Thermal Noise) 并进行抑制。

    Key Features:
    1. Cycloid Factor: phi = sqrt(L_ema / L_0) -> Adapts step size based on potential energy.
    2. Quantum Threshold (h_dl): Filters noise using Mean Action Density.
    3. Bias Correction: Standard AdamW bias correction.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        h_dl (float, optional): Quantum Threshold for action filtering (default: 1e-8)
        loss_alpha (float, optional): Smoothing factor for Loss EMA (default: 0.9)
        warmup_steps (int, optional): Steps to disable physics logic (default: 500)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, h_dl=1e-8, loss_alpha=0.9, warmup_steps=500):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        h_dl=h_dl, loss_alpha=loss_alpha, warmup_steps=warmup_steps)
        super(CycloAdamW, self).__init__(params, defaults)
        
        self.state['global_step'] = 0
        self.state['loss_ema'] = None
        self.state['initial_loss'] = None

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if loss is None:
            # If no closure is given, we cannot calculate Cycloid Factor
            # Fallback to standard AdamW step without scaling? 
            # Ideally CycloOptimizer requires Loss.
            return loss

        # --- Update Global State (Loss EMA) ---
        # 1. 追踪势能变化 (Tracking Potential Energy)
        # 我们使用 Loss 的指数移动平均 (EMA) 来代表当前的“势能高度”。
        # Use Exponential Moving Average of Loss to represent current "Potential Energy".
        curr_loss = loss.item()
        self.state['global_step'] += 1
        global_step = self.state['global_step']
        
        loss_alpha = self.defaults['loss_alpha']
        
        if self.state['loss_ema'] is None:
            self.state['loss_ema'] = curr_loss
            self.state['initial_loss'] = curr_loss
        else:
            self.state['loss_ema'] = loss_alpha * self.state['loss_ema'] + (1 - loss_alpha) * curr_loss
        
        # --- Calculate Cycloid Factor (V2 Logic) ---
        # 2. 计算摆线速度因子 (Calculate Cycloid Factor)
        # 根据最速降线原理 (Brachistochrone)，速度 v 与高度的平方根成正比。
        # v ~ sqrt(height) -> StepSize ~ sqrt(Loss)
        warmup_steps = self.defaults['warmup_steps']
        
        if global_step <= warmup_steps:
            phi = 1.0
            # V2: Continuously update initial_loss during warmup to lock in the "stable" start point
            if global_step == warmup_steps:
                self.state['initial_loss'] = self.state['loss_ema']
        else:
            # V2: Fixed L0 reference (No dynamic reset)
            # This provides a stable absolute potential energy frame
            
            # Avoid division by zero
            denom = self.state['initial_loss'] + 1e-8
            
            # V2: Standard Brachistochrone Law: v ~ sqrt(height)
            # 标准最速降线公式: 速度因子 phi = sqrt(当前势能 / 初始势能)
            # No gamma tuning.
            phi = math.sqrt(max(0, self.state['loss_ema']) / denom)
            
            # Clamp to prevent explosion
            # 限制因子范围，防止数值爆炸
            phi = max(0.1, min(phi, 1.2))

        # --- Parameter Updates ---
        for group in self.param_groups:
            h_dl = group['h_dl']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('CycloAdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1

                # 1. Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # 2. AdamW Moment Updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. Compute Naive Step Direction (Standard AdamW Step)
                # Bias Correction (Added in V2.1/V2-Best)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Standard AdamW step size with bias correction
                step_size = group['lr'] * phi * math.sqrt(bias_correction2) / bias_correction1
                
                # Check Quantum Threshold if not in warmup
                if global_step > warmup_steps and h_dl > 0:
                    # 3. 量子阈值过滤 (Quantum Threshold / Action Principle)
                    # V2: Mean Action Density (Scale Invariant)
                    # 计算“平均作用量密度”: 类似于物理中的 Action = Energy * Time (这里是 Force * Distance)
                    # We need to estimate the "unscaled" update magnitude
                    # naive_update = step_size * (exp_avg / denom) <-- This is the actual update vector
                    
                    # Work Density = Force * Distance = Grad * Update
                    # 功密度 = 力 (Gradient) * 距离 (Update Step)
                    # But we use the "Naive" update (before scaling) to check if it's worth taking
                    
                    numerator = exp_avg
                    scaled_grad = numerator / denom
                    
                    # Work = |Grad * Step|
                    # 计算每个参数做的“功” (Work)
                    work_density = scaled_grad * grad
                    
                    # Mean Action = Mean(Work) * StepSize
                    # 平均作用量 = 平均功 * 步长
                    # Note: We use p.numel() to make it scale invariant (Density)
                    mean_action = step_size * work_density.norm(p=1) / p.numel()
                    
                    if mean_action < h_dl:
                        # Soft Gating / Linear Suppression
                        # 如果平均作用量小于阈值 h_dl (普朗克常数)，则视为噪声。
                        # 使用线性抑制 (Soft Gating) 减小步长。
                        # If action is 0.5 * h_dl, we suppress step by 0.5
                        suppression = mean_action / (h_dl + 1e-12)
                        step_size = step_size * suppression.item() # Scalar mult
                
                # 4. Final Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
