"""
MNIST activation distribution analysis with monatq.

Trains a small CNN on MNIST, then hooks into the output of the first
conv block and feeds every test-set activation into a TensorDigest.
Call visualize() at the end to explore how values are distributed at
each spatial position and channel.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from monatq_py import TensorDigest

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Block 1  — output shape: [B, 8, 13, 13] after pool
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # [B,  8, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B,  8, 14, 14]
        )
        # Block 2  — output shape: [B, 16, 5, 5] after pool
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1), # [B, 16, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # [B, 16,  7,  7]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("~/.cache/mnist", train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST("~/.cache/mnist", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- Train ---
    epochs = 5
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, criterion, optimizer, device)
        acc  = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs}  loss={loss:.4f}  test_acc={acc:.4f}")

    # --- Hook TensorDigest into the first Conv2d output ---
    # Conv output shape: [B, 8, 28, 28]
    # We track each (channel, row, col) position independently → shape [8, 28, 28].
    HOOK_SHAPE = [8, 28, 28]
    td = TensorDigest(HOOK_SHAPE, compression=100)

    activations_seen = 0

    def hook_fn(module, input, output):
        nonlocal activations_seen
        # output: [B, 8, 28, 28]  float32, on device
        batch = output.detach().cpu().contiguous().float()
        for i in range(batch.shape[0]):
            td.update(batch[i])          # [8, 28, 28] — one sample
        activations_seen += batch.shape[0]

    handle = model.block1[0].register_forward_hook(hook_fn)

    # --- Inference over the full test set ---
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            model(images)

    handle.remove()
    print(f"\nCollected {activations_seen} activation samples (shape {HOOK_SHAPE})")

    # --- Visualize ---
    print("Launching monatq visualizer at http://localhost:7777 — press Ctrl+C to stop")
    td.visualize()


if __name__ == "__main__":
    main()
