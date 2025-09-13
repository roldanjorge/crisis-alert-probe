# %%
import torch
from transformer_lens.utils import get_device
from tqdm import tqdm
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader
import sklearn
from probes import Probe
from train import train_layer
from report import ReportGenerator
import pickle
import os
import pandas as pd

# %% Set device
device = get_device()
print(f"Device: {device}")

# %%
# To reload the dataset later, use:
dataset_path = (
    "/teamspace/studios/this_studio/mech_interp_exploration/data/dataset_v1.pkl"
)
with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)

# %% Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_idx, val_idx = sklearn.model_selection.train_test_split(
    # list(range(len(dataset))),
    list(range(len(dataset))),
    test_size=test_size,
    train_size=train_size,
    random_state=12345,
    shuffle=True,
    stratify=dataset.labels,
)
print(f"Train size: {train_size}, Val size: {test_size}")

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, val_idx)

sampler = None
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    sampler=sampler,
    pin_memory=True,
    batch_size=200,
    num_workers=1,
)
test_loader = DataLoader(
    test_dataset, shuffle=False, pin_memory=True, batch_size=400, num_workers=1
)

# %% Create checkpoint directory
checkpoint_dir = "probe_checkpoints/reading_probe"
os.makedirs(checkpoint_dir, exist_ok=False)

# %% Initialize metrics tracking
metrics_data = []

# %%
# Training configuration
max_epoch = 50
verbosity = True

for layer in tqdm(range(0, 40)):
    probe = Probe(num_classes=len(dataset.label2id), device=device)
    optimizer, scheduler = probe.configure_optimizers()
    best_acc = 0

    # Train the probe for this layer
    best_acc, metrics_data = train_layer(
        probe=probe,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        layer=layer,
        dataset=dataset,
        device=device,
        max_epoch=max_epoch,
        checkpoint_dir=checkpoint_dir,
        metrics_data=metrics_data,
        best_acc=best_acc,
    )

    # Save final probe state for this layer
    torch.save(probe.state_dict(), f"{checkpoint_dir}/probe_at_layer_{layer}_final.pth")
    print(f"Layer {layer} completed. Best accuracy: {best_acc:.4f}")

# %% Generate Reports and Save Metrics
print("\n" + "=" * 50)
print("GENERATING REPORTS AND SAVING METRICS")
print("=" * 50)

# Create DataFrame from metrics data
metrics_df = pd.DataFrame(metrics_data)

# Save to CSV in checkpoint directory
csv_path = os.path.join(checkpoint_dir, "probe_training_metrics.csv")
metrics_df.to_csv(csv_path, index=False)
print(f"Metrics saved to: {csv_path}")

# Create ReportGenerator instance
report_generator = ReportGenerator(
    metrics_df=metrics_df,
    dataset=dataset,
    train_size=train_size,
    test_size=test_size,
    max_epoch=max_epoch,
    checkpoint_dir=checkpoint_dir,
)

# Generate and save markdown report in checkpoint directory
markdown_path = os.path.join(checkpoint_dir, "probe_training_summary.md")
markdown_path = report_generator.save_report(markdown_path)
print(f"Markdown summary saved to: {markdown_path}")

# Print summary to console
stats, layer_summary = report_generator.print_summary()
# %%
