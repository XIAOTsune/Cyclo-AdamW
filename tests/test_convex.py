import torch
import math
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from cyclo_adamw import CycloAdamW

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def test_rosenbrock(optimizer_cls, name="CycloAdamW", steps=2000):
    print(f"\n--- Testing {name} on Rosenbrock Function ---")
    
    # Starting point: (-1.2, 1.0) is standard hard start
    x = torch.tensor([-1.2], requires_grad=True)
    y = torch.tensor([1.0], requires_grad=True)
    
    if name == "CycloAdamW":
        optimizer = optimizer_cls([x, y], lr=1e-3, weight_decay=0, h_dl=1e-5, warmup_steps=100)
    else:
        optimizer = optimizer_cls([x, y], lr=1e-3, weight_decay=0)
        
    history = []
    
    for i in range(steps):
        def closure():
            optimizer.zero_grad()
            loss = rosenbrock(x, y)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        history.append(loss.item())
        
        if i % 200 == 0:
            print(f"Step {i}: Loss = {loss.item():.6f}, x = {x.item():.4f}, y = {y.item():.4f}")
            
    final_loss = history[-1]
    print(f"Final Loss: {final_loss:.8f}")
    print(f"Final Position: ({x.item():.4f}, {y.item():.4f})")
    
    # Check if close to (1, 1)
    dist = math.sqrt((x.item() - 1)**2 + (y.item() - 1)**2)
    print(f"Distance to optimum (1, 1): {dist:.6f}")
    return history

if __name__ == "__main__":
    print("Running Convex Optimization Tests...")
    
    # Test Baseline AdamW
    test_rosenbrock(torch.optim.AdamW, name="AdamW")
    
    # Test CycloAdamW
    test_rosenbrock(CycloAdamW, name="CycloAdamW")
