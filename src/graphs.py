"""
Graphs module for different plots and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def plot_val_accuracy_vs_layer(metrics_file_path, output_dir="visualizations"):
    """
    Create a plot of validation accuracy vs layer number.
    Shows the maximum validation accuracy achieved by each layer across all epochs.
    
    Args:
        metrics_file_path (str): Path to the probe_training_metrics.csv file
        output_dir (str): Directory to save the plot (default: "visualizations")
    """
    # Read the CSV file
    df = pd.read_csv(metrics_file_path)
    
    # Remove the header row if it exists in the data
    df = df[df['layer'] != 'layer']
    
    # Convert columns to numeric
    df['layer'] = pd.to_numeric(df['layer'])
    df['epoch'] = pd.to_numeric(df['epoch'])
    df['val_accuracy'] = pd.to_numeric(df['val_accuracy'])
    
    # Filter out rows with zero validation accuracy (since validation is only computed every 10 epochs)
    df_nonzero = df[df['val_accuracy'] > 0].copy()
    
    # For each layer, find the maximum validation accuracy achieved
    layer_max_val = df_nonzero.groupby('layer')['val_accuracy'].max().reset_index()
    layer_max_val = layer_max_val.sort_values('layer')
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(layer_max_val['layer'], layer_max_val['val_accuracy'], 
             marker='o', linewidth=2, markersize=8, alpha=0.8, color='steelblue')
    
    plt.xlabel('Layer Number', fontsize=12)
    plt.ylabel('Maximum Validation Accuracy', fontsize=12)
    plt.title('Maximum Validation Accuracy Achieved by Each Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add some styling and annotations
    plt.xticks(range(0, int(layer_max_val['layer'].max()) + 1, 5))  # Show every 5th layer
    plt.ylim(0, 1.05)  # Set y-axis from 0 to 1.05 to show full range
    
    # Add text annotation for the best performing layer
    best_layer_idx = layer_max_val['val_accuracy'].idxmax()
    best_layer = layer_max_val.loc[best_layer_idx, 'layer']
    best_acc = layer_max_val.loc[best_layer_idx, 'val_accuracy']
    plt.annotate(f'Best: Layer {best_layer}\nAcc: {best_acc:.3f}', 
                xy=(best_layer, best_acc), 
                xytext=(best_layer + 5, best_acc + 0.05),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'val_accuracy_vs_layer.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")
    print(f"Best performing layer: {best_layer} with accuracy: {best_acc:.4f}")
    
    return layer_max_val


def plot_simple_val_accuracy_vs_layer(metrics_file_path, output_dir="visualizations"):
    """
    Create a simple, clean plot of validation accuracy vs layer number.
    Shows only the maximum validation accuracy achieved by each layer.
    
    Args:
        metrics_file_path (str): Path to the probe_training_metrics.csv file
        output_dir (str): Directory to save the plot (default: "visualizations")
    """
    # Read the CSV file
    df = pd.read_csv(metrics_file_path)
    
    # Remove the header row if it exists in the data
    df = df[df['layer'] != 'layer']
    
    # Convert columns to numeric
    df['layer'] = pd.to_numeric(df['layer'])
    df['val_accuracy'] = pd.to_numeric(df['val_accuracy'])
    
    # Filter out rows with zero validation accuracy
    df_nonzero = df[df['val_accuracy'] > 0].copy()
    
    # For each layer, find the maximum validation accuracy achieved
    layer_max_val = df_nonzero.groupby('layer')['val_accuracy'].max().reset_index()
    layer_max_val = layer_max_val.sort_values('layer')
    
    # Create a simple, clean plot
    plt.figure(figsize=(14, 8))
    
    # Create the main plot
    plt.plot(layer_max_val['layer'], layer_max_val['val_accuracy'], 
             marker='o', linewidth=3, markersize=8, 
             color='#2E86AB', alpha=0.8, markerfacecolor='#A23B72', 
             markeredgecolor='#2E86AB', markeredgewidth=2)
    
    # Styling
    plt.xlabel('Layer Number', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy', fontsize=14, fontweight='bold')
    plt.title('Validation Accuracy by Layer', fontsize=16, fontweight='bold', pad=20)
    
    # Grid and axis styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.05)
    plt.xlim(-1, layer_max_val['layer'].max() + 1)
    
    # Set x-axis ticks
    plt.xticks(range(0, int(layer_max_val['layer'].max()) + 1, 5), fontsize=12)
    plt.yticks(fontsize=12)
    
    # Highlight the best performing layer
    best_layer_idx = layer_max_val['val_accuracy'].idxmax()
    best_layer = layer_max_val.loc[best_layer_idx, 'layer']
    best_acc = layer_max_val.loc[best_layer_idx, 'val_accuracy']
    
    # Add annotation for best layer
    plt.annotate(f'Best Layer: {best_layer}\nAccuracy: {best_acc:.3f}', 
                xy=(best_layer, best_acc), 
                xytext=(best_layer + 8, best_acc + 0.1),
                arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2),
                fontsize=12, ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#F18F01', alpha=0.7))
    
    # Add some statistics text
    mean_acc = layer_max_val['val_accuracy'].mean()
    std_acc = layer_max_val['val_accuracy'].std()
    plt.text(0.02, 0.98, f'Mean: {mean_acc:.3f}\nStd: {std_acc:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'simple_val_accuracy_vs_layer.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Simple validation accuracy plot saved to: {output_path}")
    print(f"Best performing layer: {best_layer} with accuracy: {best_acc:.4f}")
    print(f"Mean accuracy across all layers: {mean_acc:.4f}")
    
    return layer_max_val


def plot_val_accuracy_over_epochs_by_layer(metrics_file_path, output_dir="visualizations"):
    """
    Create a plot showing validation accuracy over epochs for different layers.
    Only shows epochs where validation accuracy > 0.
    
    Args:
        metrics_file_path (str): Path to the probe_training_metrics.csv file
        output_dir (str): Directory to save the plot (default: "visualizations")
    """
    # Read the CSV file
    df = pd.read_csv(metrics_file_path)
    
    # Remove the header row if it exists in the data
    df = df[df['layer'] != 'layer']
    
    # Convert columns to numeric
    df['layer'] = pd.to_numeric(df['layer'])
    df['epoch'] = pd.to_numeric(df['epoch'])
    df['val_accuracy'] = pd.to_numeric(df['val_accuracy'])
    
    # Filter out rows with zero validation accuracy
    df_nonzero = df[df['val_accuracy'] > 0].copy()
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Get unique layers and create a color map
    unique_layers = sorted(df_nonzero['layer'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_layers)))
    
    for i, layer in enumerate(unique_layers):
        layer_data = df_nonzero[df_nonzero['layer'] == layer]
        plt.plot(layer_data['epoch'], layer_data['val_accuracy'], 
                marker='o', linewidth=2, markersize=6, 
                color=colors[i], alpha=0.8, label=f'Layer {layer}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Accuracy', fontsize=12)
    plt.title('Validation Accuracy Over Epochs by Layer', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'val_accuracy_over_epochs_by_layer.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {output_path}")
    
    return df_nonzero


def plot_training_metrics_overview(metrics_file_path, output_dir="visualizations"):
    """
    Create an overview plot showing training progress across layers and epochs.
    
    Args:
        metrics_file_path (str): Path to the probe_training_metrics.csv file
        output_dir (str): Directory to save the plot (default: "visualizations")
    """
    # Read the CSV file
    df = pd.read_csv(metrics_file_path)
    
    # Remove the header row if it exists in the data
    df = df[df['layer'] != 'layer']
    
    # Convert columns to numeric
    df['layer'] = pd.to_numeric(df['layer'])
    df['epoch'] = pd.to_numeric(df['epoch'])
    df['val_accuracy'] = pd.to_numeric(df['val_accuracy'])
    df['train_accuracy'] = pd.to_numeric(df['train_accuracy'])
    df['train_loss'] = pd.to_numeric(df['train_loss'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Validation accuracy heatmap
    pivot_val = df.pivot_table(values='val_accuracy', index='epoch', columns='layer', aggfunc='mean')
    im1 = axes[0, 0].imshow(pivot_val.values, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Validation Accuracy Heatmap')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Epoch')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Training accuracy heatmap
    pivot_train = df.pivot_table(values='train_accuracy', index='epoch', columns='layer', aggfunc='mean')
    im2 = axes[0, 1].imshow(pivot_train.values, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('Training Accuracy Heatmap')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Epoch')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Average validation accuracy per layer
    layer_avg = df.groupby('layer')['val_accuracy'].mean()
    axes[1, 0].bar(layer_avg.index, layer_avg.values, alpha=0.7)
    axes[1, 0].set_title('Average Validation Accuracy per Layer')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Average Val Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training loss over epochs for different layers (sample)
    sample_layers = df['layer'].unique()[::5]  # Every 5th layer
    for layer in sample_layers:
        layer_data = df[df['layer'] == layer]
        axes[1, 1].plot(layer_data['epoch'], layer_data['train_loss'], 
                       label=f'Layer {layer}', alpha=0.7)
    axes[1, 1].set_title('Training Loss Over Epochs (Sample Layers)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Training Loss')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'training_metrics_overview.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Overview plot saved to: {output_path}")


def plot_layer_performance_comparison(metrics_file_path, output_dir="visualizations"):
    """
    Create a comparison plot showing performance metrics across different layers.
    
    Args:
        metrics_file_path (str): Path to the probe_training_metrics.csv file
        output_dir (str): Directory to save the plot (default: "visualizations")
    """
    # Read the CSV file
    df = pd.read_csv(metrics_file_path)
    
    # Remove the header row if it exists in the data
    df = df[df['layer'] != 'layer']
    
    # Convert columns to numeric
    df['layer'] = pd.to_numeric(df['layer'])
    df['val_accuracy'] = pd.to_numeric(df['val_accuracy'])
    df['train_accuracy'] = pd.to_numeric(df['train_accuracy'])
    df['train_loss'] = pd.to_numeric(df['train_loss'])
    
    # Separate data for validation accuracy (only non-zero values) and training metrics (all values)
    df_val_nonzero = df[df['val_accuracy'] > 0].copy()
    
    # Calculate statistics per layer for validation accuracy (only non-zero values)
    val_stats = df_val_nonzero.groupby('layer')['val_accuracy'].agg(['max', 'mean', 'std']).round(4)
    val_stats.columns = ['val_accuracy_max', 'val_accuracy_mean', 'val_accuracy_std']
    
    # Calculate statistics per layer for training metrics (all values)
    train_stats = df.groupby('layer').agg({
        'train_accuracy': ['max', 'mean', 'std'],
        'train_loss': ['min', 'mean', 'std']
    }).round(4)
    
    # Flatten training stats column names
    train_stats.columns = ['_'.join(col).strip() for col in train_stats.columns]
    
    # Combine the statistics
    layer_stats = pd.concat([val_stats, train_stats], axis=1).reset_index()
    
    # Create the plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Max validation accuracy per layer
    axes[0, 0].bar(layer_stats['layer'], layer_stats['val_accuracy_max'], 
                   alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Maximum Validation Accuracy per Layer')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Max Val Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.05)
    
    # Plot 2: Mean validation accuracy (non-zero only) with error bars
    axes[0, 1].errorbar(layer_stats['layer'], layer_stats['val_accuracy_mean'], 
                       yerr=layer_stats['val_accuracy_std'], 
                       marker='o', capsize=5, capthick=2, color='steelblue')
    axes[0, 1].set_title('Mean Validation Accuracy per Layer (±1 std)\n(Non-zero values only)')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Mean Val Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)
    
    # Plot 3: Training vs Validation accuracy comparison (using max values for fair comparison)
    axes[1, 0].plot(layer_stats['layer'], layer_stats['val_accuracy_max'], 
                   marker='o', label='Validation (Max)', linewidth=2, color='steelblue')
    axes[1, 0].plot(layer_stats['layer'], layer_stats['train_accuracy_max'], 
                   marker='s', label='Training (Max)', linewidth=2, color='orange')
    axes[1, 0].set_title('Training vs Validation Accuracy Comparison\n(Maximum values)')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)
    
    # Plot 4: Training loss statistics
    axes[1, 1].bar(layer_stats['layer'], layer_stats['train_loss_mean'], 
                   alpha=0.7, color='lightcoral', label='Mean')
    axes[1, 1].bar(layer_stats['layer'], layer_stats['train_loss_min'], 
                   alpha=0.5, color='darkred', label='Min')
    axes[1, 1].set_title('Training Loss Statistics per Layer')
    axes[1, 1].set_xlabel('Layer')
    axes[1, 1].set_ylabel('Training Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'layer_performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Layer comparison plot saved to: {output_path}")
    
    return layer_stats


if __name__ == "__main__":
    # Example usage
    metrics_file = "src/probe_checkpoints/reading_probe/probe_training_metrics.csv"
    
    # Create the simple validation accuracy plot (main focus)
    print("Creating simple validation accuracy vs layer plot...")
    simple_val_data = plot_simple_val_accuracy_vs_layer(metrics_file)
    
    # Create the original validation accuracy plot
    print("\nCreating validation accuracy vs layer plot...")
    layer_max_val = plot_val_accuracy_vs_layer(metrics_file)
    
    # Create additional validation accuracy plot
    print("\nCreating validation accuracy over epochs by layer plot...")
    val_data = plot_val_accuracy_over_epochs_by_layer(metrics_file)
    
    # Create additional visualization plots
    print("\nCreating training metrics overview...")
    plot_training_metrics_overview(metrics_file)
    
    print("\nCreating layer performance comparison...")
    layer_stats = plot_layer_performance_comparison(metrics_file)
    
    print("\nAll plots have been generated and saved to the visualizations directory!")
