import torch
from tqdm import tqdm


def test_probe(probe, test_loader, layer, device, verbose=True):
    """
    Test a probe on validation data.

    Args:
        probe: The probe model to test
        test_loader: DataLoader for test/validation data
        layer: Current layer being tested
        device: Device to test on
        verbose: Whether to print results

    Returns:
        float: Validation accuracy
    """
    probe.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            target = batch["label"].to(torch.long).to(device)
            resid = batch["resids"][:, layer, :].to(device)
            output = probe(resid)
            pred = torch.argmax(output[0], dim=1)
            val_correct += (pred == target).sum().item()
            val_total += target.size(0)

    val_accuracy = val_correct / val_total
    if verbose:
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    return val_accuracy


def test_probe_detailed(probe, test_loader, layer, device, verbose=True):
    """
    Test a probe on validation data with detailed metrics.

    Args:
        probe: The probe model to test
        test_loader: DataLoader for test/validation data
        layer: Current layer being tested
        device: Device to test on
        verbose: Whether to print results

    Returns:
        tuple: (val_accuracy, predictions, targets)
    """
    probe.eval()
    val_correct = 0
    val_total = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            target = batch["label"].to(torch.long).to(device)
            resid = batch["resids"][:, layer, :].to(device)
            output = probe(resid)
            pred = torch.argmax(output[0], dim=1)
            val_correct += (pred == target).sum().item()
            val_total += target.size(0)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())

    val_accuracy = val_correct / val_total
    if verbose:
        print(f"Validation Accuracy: {val_accuracy:.4f}")

    return val_accuracy, predictions, targets
