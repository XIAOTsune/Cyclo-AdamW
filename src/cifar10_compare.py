import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cyclo_adamw_v2 import CycloAdamW
import time
import json
import os

# Configuration
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def get_cifar10_loaders():
    print("Preparing Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download to a temporary folder or check if exists in standard location
    # Assuming user might not have it, let's put it in ./data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return trainloader, testloader

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        # CycloAdamW requires closure for loss-based step (though the file says it handles closure=None partially, 
        # it is safer to check if it needs closure or we can pass loss to step if designed... 
        # Checking cyclo_adamw_v2.py: step(closure=None) -> returns loss. 
        # Inside: if closure is None: return loss (and skips logic). 
        # So we MUST provide a closure for CycloAdamW to work properly!)
        
        if isinstance(optimizer, CycloAdamW):
            def closure():
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                return loss
            optimizer.step(closure)
            # Fetch loss again since closure re-evaluates
            # Or just use the initial loss for logging approx
        else:
            optimizer.step()
            
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    avg_loss = running_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc

def run_experiment(name, optimizer_cls, optimizer_kwargs, trainloader, testloader):
    print(f"\n--- Starting Experiment: {name} ---")
    
    # Re-initialize model for fair comparison
    # We use ResNet-18
    # Note: torchvision.models.resnet18 expects 224x224 usually, but for cifar10 (32x32) 
    # we should modify the first conv layer to avoid downsampling too much.
    # However, for standard comparison, standard resnet18 is often used, or a modified one.
    # Let's use standard but be aware 32x32 might be small. 
    # Actually, for CIFAR10, it's common to replace the first conv and remove maxpool.
    
    model = torchvision.models.resnet18(num_classes=10)
    # Modify for CIFAR-10 (32x32)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    model = model.to(DEVICE)
    
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
        "epoch_time": []
    }
    
    start_time_total = time.time()
    
    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, testloader, criterion)
        t1 = time.time()
        
        epoch_time = t1 - t0
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_time"].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time_total
    print(f"Total Training Time for {name}: {total_time:.2f}s")
    
    return history

def main():
    trainloader, testloader = get_cifar10_loaders()
    
    # 1. Run AdamW
    adamw_kwargs = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    adamw_history = run_experiment("AdamW", optim.AdamW, adamw_kwargs, trainloader, testloader)
    
    # 2. Run Cyclo-AdamW v2
    cyclo_kwargs = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "h_dl": 1e-8, "warmup_steps": 500}
    cyclo_history = run_experiment("Cyclo-AdamW v2", CycloAdamW, cyclo_kwargs, trainloader, testloader)
    
    # Save Results
    results = {
        "AdamW": adamw_history,
        "CycloAdamW_v2": cyclo_history
    }
    
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
        print("\nResults saved to comparison_results.json")

if __name__ == "__main__":
    main()
