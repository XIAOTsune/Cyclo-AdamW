import math
import torch
from torch.optim import Optimizer

class CycloAdamW(Optimizer):
    """
    Cyclo-AdamW (V2): A physics-inspired optimizer based on the Brachistochrone principle.
    
    It introduces:
    1. Cycloid Factor: Dynamic step size scaling based on "potential energy" (Loss).
    2. Quantum Threshold (h_DL): Filters out noise updates using a minimum action principle.
       (V2: Uses Mean Action Density for scale invariance).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, h_dl=1e-8, loss_alpha=0.9, warmup_steps=500, gamma=0.25):
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
                        h_dl=h_dl, loss_alpha=loss_alpha, warmup_steps=warmup_steps, gamma=gamma)
        super(CycloAdamW, self).__init__(params, defaults)
        
        self.state['global_step'] = 0
        self.state['loss_ema'] = None
        self.state['initial_loss'] = None

    @torch.no_grad()
    def step(self, closure):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        with torch.enable_grad():
            loss = closure()

        if loss is None:
            raise RuntimeError("CycloAdamW requires a closure that returns the loss")
        
        curr_loss = float(loss.item())
        
        # --- Physics State Management ---
        self.state['global_step'] += 1
        global_step = self.state['global_step']
        
        # Initialize or Update Loss EMA
        if self.state['initial_loss'] is None:
            self.state['initial_loss'] = curr_loss
            self.state['loss_ema'] = curr_loss
        
        loss_alpha = self.defaults['loss_alpha']
        self.state['loss_ema'] = loss_alpha * self.state['loss_ema'] + (1 - loss_alpha) * curr_loss
        
        # --- Calculate Cycloid Factor ---
        warmup_steps = self.defaults['warmup_steps']
        
        if global_step <= warmup_steps:
            phi = 1.0
            # Continuously update initial_loss during warmup
            if global_step == warmup_steps:
                self.state['initial_loss'] = self.state['loss_ema']
        else:
            # Dynamic Calibration: If we found a new "high", reset potential
            if self.state['loss_ema'] > self.state['initial_loss']:
                self.state['initial_loss'] = self.state['loss_ema']
                
            # Avoid division by zero
            denom = self.state['initial_loss'] + 1e-8
            ratio = max(0, self.state['loss_ema']) / denom
            
            # Energy Retention: phi = ratio ^ gamma
            # Default gamma should be tunable. For now hardcode or use default dict? 
            # Ideally this is passed in __init__, but let's assume default=0.25 if not present
            gamma = self.defaults.get('gamma', 0.25)
            
            phi = math.pow(ratio, gamma)
            
            # Clamp to prevent explosion
            phi = max(0.1, min(phi, 1.2))

        # --- Parameter Updates ---
        for group in self.param_groups:
            h_dl = group['h_dl']
            
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
                beta1, beta2 = group['betas']

                state['step'] += 1

                # 1. Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # 2. AdamW Moment Updates
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. Compute Naive Step Direction (Standard AdamW Step)
                # Bias Correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # denom = sqrt(v / (1-b2^t)) + eps
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # step_size = lr * phi * (sqrt(1-b2^t) / (1-b1^t)) ? 
                # Actually standard AdamW implementation usually groups bias correction into step size:
                # step_size = lr * math.sqrt(1 - beta2^t) / (1 - beta1^t)
                # And denom is just sqrt(exp_avg_sq) + eps * ...
                
                # Let's match PyTorch AdamW implementation style for denominator
                # denom = exp_avg_sq.sqrt().add_(group['eps']) 
                # step_size = group['lr'] * phi * math.sqrt(bias_correction2) / bias_correction1
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr'] * phi * math.sqrt(bias_correction2) / bias_correction1
                
                # Check Quantum Threshold if not in warmup
                if global_step > warmup_steps and h_dl > 0:
                    # V2: Mean Action Density
                    numerator = exp_avg
                    scaled_grad = numerator / denom
                    
                    work_density = scaled_grad * grad
                    mean_action = step_size * work_density.norm(p=1) / p.numel()
                    
                    if mean_action < h_dl:
                        # Soft Gating / Linear Suppression
                        suppression = mean_action / (h_dl + 1e-12)
                        step_size = step_size * suppression.item()
                        


                
                # 4. Final Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
