import torch
import math
from torch.optim.optimizer import Optimizer

class CycloAdamW(Optimizer):
    r"""Implements Cyclo-AdamW algorithm (V2 - Classic Physics).
    
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
            # No gamma tuning.
            phi = math.sqrt(max(0, self.state['loss_ema']) / denom)
            
            # Clamp to prevent explosion
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
                    # V2: Mean Action Density (Scale Invariant)
                    # We need to estimate the "unscaled" update magnitude
                    # naive_update = step_size * (exp_avg / denom) <-- This is the actual update vector
                    
                    # Work Density = Force * Distance = Grad * Update
                    # But we use the "Naive" update (before scaling) to check if it's worth taking
                    
                    numerator = exp_avg
                    scaled_grad = numerator / denom
                    
                    # Work = |Grad * Step|
                    work_density = scaled_grad * grad
                    
                    # Mean Action = Mean(Work) * StepSize
                    # Note: We use p.numel() to make it scale invariant (Density)
                    mean_action = step_size * work_density.norm(p=1) / p.numel()
                    
                    if mean_action < h_dl:
                        # Soft Gating / Linear Suppression
                        # If action is 0.5 * h_dl, we suppress step by 0.5
                        suppression = mean_action / (h_dl + 1e-12)
                        step_size = step_size * suppression.item() # Scalar mult
                
                # 4. Final Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
