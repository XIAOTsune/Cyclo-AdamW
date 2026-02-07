import math
import torch
from torch.optim import Optimizer

class CycloAdamW(Optimizer):
    """
    Cyclo-AdamW: A physics-inspired optimizer based on the Brachistochrone principle.
    
    It introduces:
    1. Cycloid Factor: Dynamic step size scaling based on "potential energy" (Loss).
    2. Quantum Threshold (h_DL): Filters out noise updates using a minimum action principle.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, h_dl=1e-5, loss_alpha=0.9, warmup_steps=100):
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
        # Phi = sqrt(Loss_EMA / L_0)
        # Warmup: During warmup, force phi = 1.0 (Standard AdamW behavior)
        # We also re-calibrate initial_loss at the end of warmup to avoid "shock"
        warmup_steps = self.defaults['warmup_steps']
        
        if global_step <= warmup_steps:
            phi = 1.0
            # Continuously update initial_loss during warmup to capture the "stable" start point
            if global_step == warmup_steps:
                self.state['initial_loss'] = self.state['loss_ema']
        else:
            # Avoid division by zero
            denom = self.state['initial_loss'] + 1e-8
            phi = math.sqrt(max(0, self.state['loss_ema']) / denom)
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
                # denom = sqrt(v) + eps
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # We want to check the "Action" of this potential update
                # Action S = | Delta_theta * Grad L |
                # Delta_theta_naive = - lr * phi * m / denom
                # But here we stick to the paper/design: Action = | m * g | (simplified proxy)
                # Or more accurately based on design doc: S = | step * grad |
                
                # Let's use the design doc logic:
                # naive_step = lr * phi * m / denom
                # action = norm(naive_step * grad)
                
                # Optimization: To avoid full tensor allocation for naive_step just for norm,
                # we can compute action approximation using dot product if feasible, 
                # but element-wise multiplication sum is safer for "Action" scalar.
                
                step_size = group['lr'] * phi
                
                # Check Quantum Threshold if not in warmup
                if global_step > warmup_steps and h_dl > 0:
                    # Calculate action: A = sum(|step_i * g_i|) or |step . g|?
                    # Physics: Work = Force * Displacement. Dot product.
                    # A = | sum( (step_size * m / denom) * g ) |
                    
                    # We compute the dot product term: (m * g / denom).sum()
                    # To be efficient, we can't avoid some computation.
                    # Let's do a simplified element-wise action check or global?
                    # The design implies a parameter-wise (or group-wise) scalar action.
                    # Usually "Action" is a scalar for the particle (parameter tensor).
                    
                    # Compute expected step magnitude roughly
                    # step_tensor = step_size * exp_avg / denom
                    # work = (step_tensor * grad).abs().sum() 
                    
                    # NOTE: Doing this per-parameter-tensor is standard for PyTorch optimizers
                    
                    numerator = exp_avg
                    scaled_grad = numerator / denom
                    # Work = step_size * (scaled_grad * grad).sum()
                    # This is rigorous work.
                    
                    work_density = scaled_grad * grad
                    action = step_size * work_density.norm(p=1) # L1 norm of work density
                    
                    if action < h_dl:
                        # Soft Gating / Linear Suppression
                        suppression = action / (h_dl + 1e-10)
                        step_size = step_size * suppression
                
                # 4. Final Update
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
