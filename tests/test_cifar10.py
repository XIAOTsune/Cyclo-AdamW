import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from cyclo_adamw import CycloAdamW

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv1(x)))
        # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv2(x)))
        # 8x8 -> 4x4
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def ResNet18():
    return SimpleCNN()

def train(model, device, train_loader, optimizer, epoch, log_interval=50):
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            return loss
            
        loss = optimizer.step(closure)
        total_loss += loss.item()
        
        # Simple accuracy tracking
        with torch.no_grad():
             pred = model(data).argmax(dim=1, keepdim=True)
             correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    print(f"Epoch {epoch} Training Acc: {acc:.2f}%")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc

def run_experiment(optimizer_name="CycloAdamW", epochs=5):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 stats
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    batch_size = 128
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2 if use_cuda else 0)
        
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=2 if use_cuda else 0)

    model = ResNet18().to(device)
    
    # Standard settings for ResNet
    lr = 1e-3
    weight_decay = 1e-4 # Standard for ResNet/CIFAR
    
    if optimizer_name == "CycloAdamW":
        # Cyclo-AdamW (V3 Engineering: h_dl=1e-8, gamma=0.25)
        optimizer = CycloAdamW(model.parameters(), lr=lr, weight_decay=weight_decay, 
                               h_dl=1e-8, warmup_steps=200, gamma=0.25)
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    print(f"\nTraining ResNet18 on CIFAR-10 with {optimizer_name}...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        
    print(f"Total time for {optimizer_name}: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # We reduce epochs to 3 for quick verification on CPU, 
    # but still enough to see initial divergence.
    # Set to 5 if GPU is available or user is patient.
    epochs = 3
    
    print("--- Running Comparison ---")
    run_experiment("AdamW", epochs=epochs)
    run_experiment("CycloAdamW", epochs=epochs)
