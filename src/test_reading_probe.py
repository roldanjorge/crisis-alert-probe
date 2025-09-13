#!/usr/bin/env python3
"""
Test script for evaluating the reading probe performance on mood dataset.

This script implements a ProbeTester class that:
1. Loads the mood dataset from CSV
2. Loads the LLaMA-2 model and reading probe 39
3. Runs inference on all test cases
4. Computes evaluation metrics and creates visualizations
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import os
from pathlib import Path
from tqdm import tqdm
import warnings

# Import local modules
from model import get_model
from probes import Probe
from utils import llama_v2_prompt

warnings.filterwarnings("ignore")

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ProbeTester:
    """Class for testing reading probe performance on mood dataset."""
    
    def __init__(self, 
                 data_path="/teamspace/studios/this_studio/mech_interp_exploration/data/mood_dataset_2100.csv",
                 probe_layer=30,
                 probe_checkpoint_dir="/teamspace/studios/this_studio/mech_interp_exploration/src/probe_checkpoints/reading_probe"):
        """
        Initialize the ProbeTester.
        
        Args:
            data_path: Path to the mood dataset CSV file
            probe_layer: Layer number for the probe to test (default: 39)
            probe_checkpoint_dir: Directory containing probe checkpoints
        """
        self.data_path = data_path
        self.probe_layer = probe_layer
        self.probe_checkpoint_dir = probe_checkpoint_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define label mapping (same as training)
        self.label2id = {
            "very_happy": 0,
            "happy": 1,
            "slightly_positive": 2,
            "neutral": 3,
            "slightly_negative": 4,
            "sad": 5,
            "very_sad": 6,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # Initialize attributes
        self.model = None
        self.probe = None
        self.test_cases = None
        self.results = None
        
        print(f"Initializing ProbeTester on device: {self.device}")
        print(f"Probe layer: {probe_layer}")
        print(f"Data path: {data_path}")
        
    def load_data(self):
        """Load mood dataset from CSV into test_cases dataframe."""
        print("Loading mood dataset...")
        self.test_cases = pd.read_csv(self.data_path, header=0)
        print(f"Loaded {len(self.test_cases)} test cases")
        print(f"Dataset columns: {list(self.test_cases.columns)}")
        print(f"Label distribution:")
        print(self.test_cases['mood'].value_counts().sort_index())
        
    def load_model(self):
        """Load the LLaMA-2 model using get_model() from model.py."""
        print("Loading LLaMA-2 model...")
        self.model = get_model()
        print(f"Model loaded successfully: {self.model.cfg}")
        
    def load_probe(self):
        """Load the reading probe for the specified layer."""
        print(f"Loading reading probe for layer {self.probe_layer}...")
        
        # Create probe instance
        self.probe = Probe(num_classes=len(self.label2id), device=self.device)
        
        # Try to load the best checkpoint first, then fall back to final
        checkpoint_paths = [
            f"{self.probe_checkpoint_dir}/probe_at_layer_{self.probe_layer}.pth",  # Best checkpoint
            f"{self.probe_checkpoint_dir}/probe_at_layer_{self.probe_layer}_final.pth"  # Final checkpoint
        ]
        
        loaded = False
        for path in checkpoint_paths:
            try:
                self.probe.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Loaded probe from {path}")
                loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not loaded:
            raise FileNotFoundError(f"No checkpoint found for layer {self.probe_layer}")
        
        self.probe.eval()
        print(f"Probe loaded successfully for layer {self.probe_layer}")
        
    def extract_activations(self, text):
        """
        Extract residual activations from LLaMA-2 for the given text.
        
        Args:
            text: Input text to process
            
        Returns:
            Residual activation for the probe layer
        """
        # Format the text with the same prompt structure used in training
        messages = [{"content": text, "role": "user"}]
        formatted_text = llama_v2_prompt(messages)
        formatted_text += " I think the emotional state of this user is"
        
        with torch.no_grad():
            tokens = self.model.to_tokens(formatted_text)
            
            # Ensure sequence length constraint (same as training)
            if tokens.shape[-1] > 2048:
                tokens = tokens[:, -2048:]
            
            # Run model with cache to get activations
            _, cache = self.model.run_with_cache(
                tokens.to(self.device), remove_batch_dim=False
            )
            
            # Extract residual activation for the probe layer (last token)
            resid_post = cache["resid_post", self.probe_layer][:, -1].detach().cpu().to(torch.float)
            
            # Clean up to save memory
            del tokens, cache
            torch.cuda.empty_cache()
            
            return resid_post
    
    def run(self):
        """
        Run the probe tester on all test cases.
        Iterates over each row in test_cases and stores results.
        """
        print("Starting probe testing...")
        
        # Initialize results storage
        results_data = []
        
        # Process each test case
        for idx in tqdm(range(len(self.test_cases)), desc="Processing test cases"):
            try:
                # Get test case data
                case_id = self.test_cases.iloc[idx]['case_id']
                message = self.test_cases.iloc[idx]['message']
                true_mood = self.test_cases.iloc[idx]['mood']
                true_label = self.label2id[true_mood]
                
                # Extract activations using model.run_with_cache
                resid_post = self.extract_activations(message)
                
                # Use the reading probe to get mood output
                with torch.no_grad():
                    resid_post = resid_post.to(self.device)
                    output = self.probe(resid_post)
                    probabilities = output[0].cpu().numpy()[0]
                    predicted_class = torch.argmax(output[0], dim=1).cpu().item()
                    predicted_mood = self.id2label[predicted_class]
                    confidence = float(probabilities[predicted_class])
                
                # Store results
                result = {
                    'case_id': case_id,
                    'message': message,
                    'true_mood': true_mood,
                    'true_label': true_label,
                    'predicted_mood': predicted_mood,
                    'predicted_label': predicted_class,
                    'confidence': confidence,
                    'correct': true_label == predicted_class,
                    'all_probabilities': probabilities.tolist()
                }
                
                # Add individual probability scores
                for i, mood in self.id2label.items():
                    result[f'prob_{mood}'] = float(probabilities[i])
                
                results_data.append(result)
                
            except Exception as e:
                print(f"Error processing case {idx}: {e}")
                continue
        
        # Convert to DataFrame
        self.results = pd.DataFrame(results_data)
        
        print(f"Completed testing on {len(self.results)} cases")
        print(f"Overall accuracy: {self.results['correct'].mean():.4f}")
        
        return self.results
    
    def compute_metrics(self):
        """Compute confusion matrix and classification report using scikit-learn."""
        if self.results is None:
            raise ValueError("No results available. Run the test first.")
        
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        # Extract true and predicted labels
        y_true = self.results['true_label'].values
        y_pred = self.results['predicted_label'].values
        
        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Get unique labels present in the data
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        unique_label_names = [self.id2label[label] for label in unique_labels]
        
        print(f"Classes present in data: {unique_label_names}")
        print(f"Number of unique classes: {len(unique_labels)}")
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        # Compute classification report with only present classes
        class_report = classification_report(
            y_true, y_pred, 
            labels=unique_labels,
            target_names=unique_label_names,
            output_dict=True
        )
        
        # Compute precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        # Print per-class metrics
        print("\nPer-class metrics:")
        for mood in unique_label_names:
            if mood in class_report:
                metrics = class_report[mood]
                print(f"{mood:20s}: Precision={metrics['precision']:.3f}, "
                      f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'unique_labels': unique_labels,
            'unique_label_names': unique_label_names
        }
    
    def create_visualizations(self, save_dir="probe_test_results"):
        """Create beautiful and insightful graphs."""
        if self.results is None:
            raise ValueError("No results available. Run the test first.")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Creating visualizations in {save_dir}...")
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # 1. Confusion Matrix Heatmap
        self._plot_confusion_matrix(save_dir)
        
        # 2. Accuracy by Mood Class
        self._plot_accuracy_by_class(save_dir)
        
        # 3. Confidence Distribution
        self._plot_confidence_distribution(save_dir)
        
        # 4. Prediction Confidence vs Accuracy
        self._plot_confidence_vs_accuracy(save_dir)
        
        # 5. Per-class Performance Metrics
        self._plot_per_class_metrics(save_dir)
        
        # 6. Error Analysis
        self._plot_error_analysis(save_dir)
        
        print(f"All visualizations saved to {save_dir}/")
    
    def _plot_confusion_matrix(self, save_dir):
        """Plot confusion matrix heatmap."""
        y_true = self.results['true_label'].values
        y_pred = self.results['predicted_label'].values
        
        # Get unique labels present in the data
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        unique_label_names = [self.id2label[label] for label in unique_labels]
        
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=unique_label_names,
                   yticklabels=unique_label_names)
        plt.title(f'Confusion Matrix - Reading Probe Layer {self.probe_layer}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Mood', fontsize=14)
        plt.ylabel('True Mood', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_accuracy_by_class(self, save_dir):
        """Plot accuracy for each mood class."""
        class_accuracy = []
        class_names = []
        
        for mood, label_id in self.label2id.items():
            class_data = self.results[self.results['true_label'] == label_id]
            if len(class_data) > 0:
                accuracy = class_data['correct'].mean()
                class_accuracy.append(accuracy)
                class_names.append(mood)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(class_names, class_accuracy, color=sns.color_palette("husl", len(class_names)))
        plt.title(f'Accuracy by Mood Class - Reading Probe Layer {self.probe_layer}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Mood Class', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/accuracy_by_class.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self, save_dir):
        """Plot distribution of prediction confidence."""
        plt.figure(figsize=(12, 6))
        
        # Overall confidence distribution
        plt.subplot(1, 2, 1)
        plt.hist(self.results['confidence'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Overall Confidence Distribution', fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Confidence by correctness
        plt.subplot(1, 2, 2)
        correct_conf = self.results[self.results['correct']]['confidence']
        incorrect_conf = self.results[~self.results['correct']]['confidence']
        
        plt.hist(correct_conf, bins=20, alpha=0.7, label='Correct', color='green')
        plt.hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', color='red')
        plt.title('Confidence by Prediction Correctness', fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_vs_accuracy(self, save_dir):
        """Plot confidence vs accuracy relationship."""
        # Bin confidence scores and compute accuracy for each bin
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        accuracies = []
        
        for i in range(len(bins)-1):
            mask = (self.results['confidence'] >= bins[i]) & (self.results['confidence'] < bins[i+1])
            if mask.sum() > 0:
                acc = self.results[mask]['correct'].mean()
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8, color='darkblue')
        plt.title(f'Confidence vs Accuracy - Reading Probe Layer {self.probe_layer}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Add trend line
        z = np.polyfit(bin_centers, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(bin_centers, p(bin_centers), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_metrics(self, save_dir):
        """Plot precision, recall, and F1-score for each class."""
        y_true = self.results['true_label'].values
        y_pred = self.results['predicted_label'].values
        
        # Get unique labels present in the data
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        unique_label_names = [self.id2label[label] for label in unique_labels]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=unique_labels
        )
        
        x = np.arange(len(unique_labels))
        width = 0.25
        
        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
        plt.bar(x, recall, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        plt.title(f'Per-Class Performance Metrics - Reading Probe Layer {self.probe_layer}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Mood Class', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.xticks(x, unique_label_names, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_analysis(self, save_dir):
        """Plot error analysis - most common misclassifications."""
        # Get incorrect predictions
        incorrect = self.results[~self.results['correct']]
        
        if len(incorrect) == 0:
            print("No errors to analyze!")
            return
        
        # Count misclassification patterns
        error_patterns = incorrect.groupby(['true_mood', 'predicted_mood']).size().reset_index(name='count')
        error_patterns = error_patterns.sort_values('count', ascending=False)
        
        # Plot top misclassifications
        plt.figure(figsize=(12, 8))
        top_errors = error_patterns.head(10)
        
        # Create labels for the plot
        labels = [f"{row['true_mood']} → {row['predicted_mood']}" 
                for _, row in top_errors.iterrows()]
        
        bars = plt.barh(labels, top_errors['count'], color='lightcoral', alpha=0.8)
        plt.title(f'Top Misclassifications - Reading Probe Layer {self.probe_layer}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Number of Cases', fontsize=14)
        plt.ylabel('True → Predicted', fontsize=14)
        
        # Add value labels
        for bar, count in zip(bars, top_errors['count']):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(count), ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, save_dir="probe_test_results"):
        """Save detailed results to CSV."""
        if self.results is None:
            raise ValueError("No results available. Run the test first.")
        
        os.makedirs(save_dir, exist_ok=True)
        /teamspace/studios/this_studio/mech_interp_exploration/data/mood_dataset_2002.csv
        # Save detailed results
        results_path = f"{save_dir}/detailed_results.csv"
        self.results.to_csv(results_path, index=False)
        print(f"Detailed results saved to {results_path}")
        
        # Save summary statistics
        summary = {
            'total_cases': len(self.results),
            'correct_predictions': self.results['correct'].sum(),
            'overall_accuracy': self.results['correct'].mean(),
            'mean_confidence': self.results['confidence'].mean(),
            'std_confidence': self.results['confidence'].std(),
            'probe_layer': self.probe_layer
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = f"{save_dir}/summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to {summary_path}")


def main():
    """Main function to run the probe test."""
    print("="*80)
    print("READING PROBE TESTER")
    print("="*80)
    
    # Initialize tester
    tester = ProbeTester(
        data_path="/teamspace/studios/this_studio/mech_interp_exploration/data/very_sad_1000_with_subcategory.csv",
        probe_layer=30
    )
    
    # Load data and models
    tester.load_data()
    tester.load_model()
    tester.load_probe()
    
    # Run tests
    results = tester.run()
    
    # Compute metrics
    metrics = tester.compute_metrics()
    
    # Create visualizations
    tester.create_visualizations()
    
    # Save results
    tester.save_results()
    
    print("\n" + "="*80)
    print("TESTING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved to: probe_test_results/")
    print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"Check the generated graphs for detailed analysis.")


if __name__ == "__main__":
    main()
