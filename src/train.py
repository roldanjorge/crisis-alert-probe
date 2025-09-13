import torch
import torch.nn as nn
from tqdm import tqdm
from probes import Probe
from test import test_probe


def train_probe(
    probe,
    train_loader,
    optimizer,
    layer,
    dataset,
    device,
    epoch,
    metrics_data,
    checkpoint_dir,
):
    """
    Train a single probe for one epoch.

    Args:
        probe: The probe model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for the probe
        layer: Current layer being trained
        dataset: Dataset object (for label2id mapping)
        device: Device to train on
        epoch: Current epoch number
        metrics_data: List to store metrics
        checkpoint_dir: Directory to save checkpoints

    Returns:
        tuple: (avg_loss, accuracy, metrics_data)
    """
    probe.train()
    loss_sum = 0
    correct = 0
    total = 0
    preds = []
    truths = []

    for batch in tqdm(train_loader):
        target = batch["label"].to(torch.long).to(device)
        optimizer.zero_grad()
        resid = batch["resids"][:, layer, :].to(device)
        output = probe(resid)

        # Convert target to one-hot encoding for BCELoss
        target_one_hot = torch.zeros(
            target.size(0), len(dataset.label2id), device=device
        )
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # Use BCELoss with one-hot targets
        loss = nn.BCELoss()(output[0], target_one_hot)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        # For prediction, use argmax on the sigmoid output
        pred = torch.argmax(output[0], dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
        preds.extend(pred.cpu().numpy())
        truths.extend(target.cpu().numpy())

    # Calculate accuracy
    accuracy = correct / total
    avg_loss = loss_sum / len(train_loader)

    return avg_loss, accuracy, preds, truths


def train_layer(
    probe,
    train_loader,
    test_loader,
    optimizer,
    layer,
    dataset,
    device,
    max_epoch,
    checkpoint_dir,
    metrics_data,
    best_acc,
):
    """
    Train a probe for multiple epochs on a single layer.

    Args:
        probe: The probe model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        optimizer: Optimizer for the probe
        layer: Current layer being trained
        dataset: Dataset object (for label2id mapping)
        device: Device to train on
        max_epoch: Maximum number of epochs
        checkpoint_dir: Directory to save checkpoints
        metrics_data: List to store metrics
        best_acc: Best accuracy achieved so far for this layer

    Returns:
        tuple: (best_acc, metrics_data)
    """
    print("-" * 40 + f"Layer {layer}" + "-" * 40)

    for epoch in range(1, max_epoch + 1):
        # Training
        avg_loss, accuracy, preds, truths = train_probe(
            probe,
            train_loader,
            optimizer,
            layer,
            dataset,
            device,
            epoch,
            metrics_data,
            checkpoint_dir,
        )
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation
        val_accuracy = 0.0  # Initialize for cases where validation doesn't run
        if epoch % 10 == 0 or epoch == max_epoch:
            val_accuracy = test_probe(probe, test_loader, layer, device, verbose=True)

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                print(f"New best accuracy: {best_acc:.4f}")
                # Save the best probe for this layer
                torch.save(
                    probe.state_dict(), f"{checkpoint_dir}/probe_at_layer_{layer}.pth"
                )

        # Store metrics for this epoch
        metrics_data.append(
            {
                "layer": layer,
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_accuracy": accuracy,
                "val_accuracy": val_accuracy,
                "best_accuracy": best_acc,
                "is_best": val_accuracy == best_acc if val_accuracy > 0 else False,
            }
        )

    return best_acc, metrics_data
