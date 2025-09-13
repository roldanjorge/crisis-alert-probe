"""
Report Generator Module

This module provides a comprehensive ReportGenerator class for creating
detailed markdown reports of probe training experiments.
"""

import pandas as pd
from datetime import datetime
import os


class ReportGenerator:
    """
    A comprehensive report generator for probe training experiments.
    
    This class generates professional markdown reports with detailed metrics,
    configuration details, and analysis of probe training results.
    
    Attributes:
        metrics_df (pd.DataFrame): Training metrics data
        dataset: Dataset object with label2id mapping
        train_size (int): Number of training samples
        test_size (int): Number of test samples
        max_epoch (int): Maximum epochs per layer
        checkpoint_dir (str): Directory where checkpoints are saved
        timestamp (str): Report generation timestamp
    """

    def __init__(
        self, 
        metrics_df, 
        dataset, 
        train_size, 
        test_size, 
        max_epoch, 
        checkpoint_dir
    ):
        """
        Initialize the ReportGenerator.

        Args:
            metrics_df (pd.DataFrame): DataFrame containing training metrics
            dataset: Dataset object with label2id mapping
            train_size (int): Number of training samples
            test_size (int): Number of test samples
            max_epoch (int): Maximum epochs per layer
            checkpoint_dir (str): Directory where checkpoints are saved
        """
        self.metrics_df = metrics_df
        self.dataset = dataset
        self.train_size = train_size
        self.test_size = test_size
        self.max_epoch = max_epoch
        self.checkpoint_dir = checkpoint_dir
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate derived statistics
        self._calculate_statistics()

    def _calculate_statistics(self):
        """
        Calculate derived statistics from metrics data.
        
        This method computes various performance metrics and identifies
        the best performing layer based on validation accuracy.
        """
        # Basic counts
        self.total_layers = self.metrics_df["layer"].nunique()
        self.total_epochs = len(self.metrics_df)
        
        # Performance metrics
        self.best_overall_acc = self.metrics_df["best_accuracy"].max()
        self.avg_train_acc = self.metrics_df["train_accuracy"].mean()
        
        # Average validation accuracy (only for epochs with validation)
        val_data = self.metrics_df[self.metrics_df["val_accuracy"] > 0]
        self.avg_val_acc = val_data["val_accuracy"].mean() if len(val_data) > 0 else 0.0

        # Find best performing layer
        best_idx = self.metrics_df["best_accuracy"].idxmax()
        self.best_layer = self.metrics_df.loc[best_idx, "layer"]
        self.best_layer_acc = self.metrics_df["best_accuracy"].max()

    # ============================================================================
    # REPORT SECTION GENERATORS
    # ============================================================================

    def generate_header(self):
        """
        Generate the report header and metadata section.
        
        Returns:
            str: Formatted markdown header with timestamp and overview
        """
        return f"""# Probe Training Summary Report

**Generated on:** {self.timestamp}

## Overview
This report summarizes the training results for linear probes on transformer 
residual streams using the One-vs-Rest (OvR) approach with BCELoss.

"""

    def generate_configuration_section(self):
        """
        Generate the training configuration section.
        
        Returns:
            str: Formatted markdown section with training hyperparameters
        """
        layer_range = f"{self.metrics_df['layer'].min()}-{self.metrics_df['layer'].max()}"
        
        return f"""## Training Configuration
- **Model:** Transformer residual streams
- **Probe Type:** Linear probe with Sigmoid activation
- **Loss Function:** BCELoss (Binary Cross Entropy)
- **Approach:** One-vs-Rest multi-class classification
- **Layers Trained:** {self.total_layers} (layers {layer_range})
- **Epochs per Layer:** {self.max_epoch}
- **Batch Size:** 200 (train), 400 (validation)
- **Learning Rate:** 1e-3
- **Weight Decay:** 0.1

"""

    def generate_dataset_section(self):
        """
        Generate the dataset information section.
        
        Returns:
            str: Formatted markdown section with dataset details
        """
        class_labels = list(self.dataset.label2id.keys())
        
        return f"""## Dataset Information
- **Total Samples:** {len(self.dataset)}
- **Training Samples:** {self.train_size}
- **Validation Samples:** {self.test_size}
- **Number of Classes:** {len(self.dataset.label2id)}
- **Class Labels:** {class_labels}

"""

    def generate_results_summary(self):
        """
        Generate the results summary section.
        
        Returns:
            str: Formatted markdown section with performance metrics
        """
        return f"""## Results Summary

### Overall Performance
- **Total Epochs Trained:** {self.total_epochs}
- **Best Overall Accuracy:** {self.best_overall_acc:.4f}
- **Average Training Accuracy:** {self.avg_train_acc:.4f}
- **Average Validation Accuracy:** {self.avg_val_acc:.4f}

### Best Performing Layer
- **Layer:** {self.best_layer}
- **Best Accuracy:** {self.best_layer_acc:.4f}

"""

    def generate_per_layer_table(self):
        """
        Generate the per-layer results table.
        
        Returns:
            str: Formatted markdown table with per-layer performance
        """
        table = """### Per-Layer Results
| Layer | Best Accuracy | Training Epochs |
|-------|---------------|-----------------|
"""

        # Sort layers and generate table rows
        for layer in sorted(self.metrics_df["layer"].unique()):
            layer_data = self.metrics_df[self.metrics_df["layer"] == layer]
            best_acc = layer_data["best_accuracy"].max()
            epochs = len(layer_data)
            table += f"| {layer} | {best_acc:.4f} | {epochs} |\n"

        return table + "\n"

    def generate_training_progress_table(self):
        """
        Generate the detailed training progress table.
        
        Returns:
            str: Formatted markdown table with epoch-by-epoch progress
        """
        table = """## Training Progress
The following table shows the detailed training progress for each epoch:

| Layer | Epoch | Train Loss | Train Acc | Val Acc | Best Acc | Is Best |
|-------|-------|------------|-----------|---------|----------|---------|
"""

        # Generate table rows for each epoch
        for _, row in self.metrics_df.iterrows():
            is_best = "✓" if row["is_best"] else ""
            table += (
                f"| {row['layer']} | {row['epoch']} | "
                f"{row['train_loss']:.4f} | {row['train_accuracy']:.4f} | "
                f"{row['val_accuracy']:.4f} | {row['best_accuracy']:.4f} | "
                f"{is_best} |\n"
            )

        return table + "\n"

    def generate_files_section(self):
        """
        Generate the files generated section.
        
        Returns:
            str: Formatted markdown section listing generated files
        """
        return f"""## Files Generated
- **Checkpoints:** `{self.checkpoint_dir}/` (probe_at_layer_*.pth files)
- **Metrics CSV:** `probe_training_metrics.csv`
- **Summary Report:** `probe_training_summary.md`

"""

    def generate_technical_details(self):
        """
        Generate the technical details section.
        
        Returns:
            str: Formatted markdown section with technical implementation details
        """
        return """## Technical Details

### Model Architecture
```python
class Probe(nn.Module):
    def __init__(self, num_classes, device):
        self.classifier = nn.Sequential(
            nn.Linear(input_dim=5120, num_classes),
            nn.Sigmoid()
        )
```

### Loss Function
- **BCELoss:** Binary Cross Entropy Loss
- **Target Format:** One-hot encoded vectors
- **Output Format:** Sigmoid probabilities for each class

### Training Strategy
1. **One-vs-Rest Approach:** Each class treated as independent binary problem
2. **Checkpoint Saving:** Best model saved when validation accuracy improves
3. **Final Model:** Last epoch model saved regardless of performance
4. **Metrics Tracking:** Comprehensive logging of all training metrics

"""

    def generate_analysis_notes(self):
        """
        Generate the analysis notes section.
        
        Returns:
            str: Formatted markdown section with technical insights
        """
        return """## Analysis Notes
- The One-vs-Rest approach with BCELoss provides independent probability 
  estimates for each class
- Sigmoid activation ensures outputs are in [0,1] range suitable for BCELoss
- Argmax is used for final prediction to select the class with highest probability
- Validation accuracy is used to determine the best model for each layer

---
*Report generated automatically by probe training script*

"""

    # ============================================================================
    # UTILITY METHODS
    # ============================================================================

    def generate_summary_statistics(self):
        """
        Generate summary statistics for console output.
        
        Returns:
            dict: Dictionary containing key performance statistics
        """
        return {
            "total_epochs": self.total_epochs,
            "total_layers": self.total_layers,
            "best_overall_acc": self.best_overall_acc,
            "avg_train_acc": self.avg_train_acc,
            "avg_val_acc": self.avg_val_acc,
            "best_layer": self.best_layer,
            "best_layer_acc": self.best_layer_acc,
        }

    def generate_per_layer_summary(self):
        """
        Generate per-layer summary for console output.
        
        Returns:
            list: List of formatted strings with per-layer performance
        """
        layer_summary = []
        
        for layer in sorted(self.metrics_df["layer"].unique()):
            layer_data = self.metrics_df[self.metrics_df["layer"] == layer]
            best_acc = layer_data["best_accuracy"].max()
            layer_summary.append(f"Layer {layer}: {best_acc:.4f}")
            
        return layer_summary

    def generate_full_report(self):
        """
        Generate the complete markdown report.
        
        Returns:
            str: Complete markdown report combining all sections
        """
        report_sections = [
            self.generate_header(),
            self.generate_configuration_section(),
            self.generate_dataset_section(),
            self.generate_results_summary(),
            self.generate_per_layer_table(),
            self.generate_training_progress_table(),
            self.generate_files_section(),
            self.generate_technical_details(),
            self.generate_analysis_notes(),
        ]

        return "".join(report_sections)

    def save_report(self, filepath="probe_training_summary.md"):
        """
        Save the complete report to a markdown file.
        
        Args:
            filepath (str): Path where to save the report file
            
        Returns:
            str: Path to the saved file
        """
        report_content = self.generate_full_report()

        with open(filepath, "w") as f:
            f.write(report_content)

        return filepath

    def print_summary(self):
        """
        Print a formatted summary to console.
        
        Returns:
            tuple: (stats_dict, layer_summary_list)
        """
        stats = self.generate_summary_statistics()
        layer_summary = self.generate_per_layer_summary()

        print("\nSummary Statistics:")
        print(f"Total epochs trained: {stats['total_epochs']}")
        print(f"Layers trained: {stats['total_layers']}")
        print(f"Best overall accuracy: {stats['best_overall_acc']:.4f}")
        print(f"Average training accuracy: {stats['avg_train_acc']:.4f}")
        print(f"Average validation accuracy: {stats['avg_val_acc']:.4f}")

        print("\nBest accuracy per layer:")
        for layer_info in layer_summary:
            print(f"  {layer_info}")

        return stats, layer_summary