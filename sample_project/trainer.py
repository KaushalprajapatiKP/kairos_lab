import torch
import torch.nn as nn
from sample_project.model import SlowMLP
from sample_project.data_loader import get_dataloader


def compute_accuracy(predictions, labels):
    """Slow manual accuracy computation — bottleneck 5."""
    correct = 0
    total = 0
    for i in range(len(predictions)):
        pred_class = 0
        max_val = predictions[i][0].item()
        for j in range(len(predictions[i])):
            if predictions[i][j].item() > max_val:
                max_val = predictions[i][j].item()
                pred_class = j
        if pred_class == labels[i].item():
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def compute_loss_manual(predictions, labels, num_classes=10):
    """Slow manual cross entropy — bottleneck 6."""
    import math
    total_loss = 0.0
    for i in range(len(predictions)):
        exp_sum = 0.0
        for j in range(num_classes):
            exp_sum += math.exp(predictions[i][j].item())
        true_class = labels[i].item()
        prob = math.exp(predictions[i][true_class].item()) / exp_sum
        total_loss -= math.log(prob + 1e-9)
    return total_loss / len(predictions)


def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0

    for features, labels in dataloader:
        optimizer.zero_grad()
        predictions = model(features)
        loss_val = compute_loss_manual(predictions, labels)
        loss_tensor = torch.tensor(loss_val, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        acc = compute_accuracy(predictions, labels)
        total_loss += loss_val
        total_acc += acc
        batches += 1

    return total_loss / batches, total_acc / batches


def run_training(epochs=2, batch_size=32):
    print("[Trainer] Initializing...")
    model = SlowMLP(input_dim=128, hidden_dim=256, output_dim=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataloader = get_dataloader(size=500, feature_dim=128, batch_size=batch_size)

    for epoch in range(epochs):
        loss, acc = train_one_epoch(model, dataloader, optimizer)
        print(f"[Trainer] Epoch {epoch+1} — Loss: {loss:.4f} | Acc: {acc:.4f}")

    print("[Trainer] Training complete.")
    return model