#!/usr/bin/env python3
"""
Inference script for using trained emotional state probes with LLaMA-2.

This script demonstrates how to:
1. Load a trained probe from checkpoint
2. Run inference on new text using transformer_lens
3. Extract emotional state predictions from LLaMA-2 activations

Usage:
    python inference_with_probes.py --text "Your text here" --layer 39
"""

import torch
import argparse
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from probes import Probe
from utils import llama_v2_prompt
import pandas as pd


class EmotionalStateInference:
    """Class for running emotional state inference using trained probes."""
    
    def __init__(self, model_name="meta-llama/Llama-2-13b-chat-hf", checkpoint_dir="probe_checkpoints/reading_probe"):
        """
        Initialize the inference system.
        
        Args:
            model_name: Name of the LLaMA-2 model to load
            checkpoint_dir: Directory containing trained probe checkpoints
        """
        self.device = get_device()
        self.checkpoint_dir = checkpoint_dir
        
        # Load the LLaMA-2 model
        print(f"Loading model: {model_name}")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=self.device,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
        )
        
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
        
        # Load training metrics to find best layers
        self.metrics_df = pd.read_csv(f"{checkpoint_dir}/probe_training_metrics.csv")
        self.best_layers = self.metrics_df.groupby('layer')['best_accuracy'].max().sort_values(ascending=False)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Top 5 performing layers: {self.best_layers.head().to_dict()}")
    
    def load_probe(self, layer):
        """
        Load a trained probe for the specified layer.
        
        Args:
            layer: Layer number to load probe for
            
        Returns:
            Loaded probe model
        """
        probe = Probe(num_classes=len(self.label2id), device=self.device)
        
        # Try to load the best checkpoint first, then fall back to final
        checkpoint_paths = [
            f"{self.checkpoint_dir}/probe_at_layer_{layer}.pth",  # Best checkpoint
            f"{self.checkpoint_dir}/probe_at_layer_{layer}_final.pth"  # Final checkpoint
        ]
        
        loaded = False
        for path in checkpoint_paths:
            try:
                probe.load_state_dict(torch.load(path, map_location=self.device))
                print(f"Loaded probe for layer {layer} from {path}")
                loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not loaded:
            raise FileNotFoundError(f"No checkpoint found for layer {layer}")
        
        probe.eval()
        return probe
    
    def extract_activations(self, text):
        """
        Extract residual activations from LLaMA-2 for the given text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing activations for each layer
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
            
            # Extract residual activations for each layer
            activations = {}
            for layer in range(self.model.cfg.n_layers):
                resid_post = cache["resid_post", layer][:, -1].detach().cpu().to(torch.float)
                activations[layer] = resid_post
            
            return activations
    
    def predict_emotional_state(self, text, layer=None):
        """
        Predict emotional state for the given text using a trained probe.
        
        Args:
            text: Input text to analyze
            layer: Layer to use for prediction (if None, uses best performing layer)
            
        Returns:
            Dictionary containing prediction results
        """
        if layer is None:
            layer = self.best_layers.index[0]  # Use best performing layer
        
        print(f"Analyzing text with probe trained on layer {layer}")
        print(f"Text: '{text}'")
        
        # Load the probe for the specified layer
        probe = self.load_probe(layer)
        
        # Extract activations
        activations = self.extract_activations(text)
        
        # Get activation for the specified layer
        layer_activation = activations[layer].to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = probe(layer_activation)
            probabilities = output[0].cpu().numpy()[0]  # Get probabilities for first (and only) sample
            predicted_class = torch.argmax(output[0], dim=1).cpu().item()
        
        # Format results
        results = {
            "text": text,
            "layer": layer,
            "predicted_emotion": self.id2label[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "all_probabilities": {
                self.id2label[i]: float(prob) for i, prob in enumerate(probabilities)
            }
        }
        
        return results
    
    def compare_layers(self, text, top_k=5):
        """
        Compare predictions across the top-k performing layers.
        
        Args:
            text: Input text to analyze
            top_k: Number of top layers to compare
            
        Returns:
            List of prediction results for each layer
        """
        results = []
        top_layers = self.best_layers.head(top_k).index.tolist()
        
        print(f"Comparing predictions across top {top_k} layers: {top_layers}")
        
        for layer in top_layers:
            try:
                result = self.predict_emotional_state(text, layer)
                results.append(result)
            except Exception as e:
                print(f"Error with layer {layer}: {e}")
                continue
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Emotional state inference using trained probes")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--layer", type=int, help="Layer to use (if not specified, uses best layer)")
    parser.add_argument("--compare", action="store_true", help="Compare across top 5 layers")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-13b-chat-hf", 
                       help="Model name to load")
    
    args = parser.parse_args()
    
    # Initialize inference system
    inference = EmotionalStateInference(model_name=args.model)
    
    if args.compare:
        # Compare across multiple layers
        results = inference.compare_layers(args.text, top_k=5)
        
        print("\n" + "="*80)
        print("COMPARISON ACROSS TOP LAYERS")
        print("="*80)
        
        for result in results:
            print(f"\nLayer {result['layer']}:")
            print(f"  Predicted Emotion: {result['predicted_emotion']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  All Probabilities:")
            for emotion, prob in result['all_probabilities'].items():
                print(f"    {emotion}: {prob:.4f}")
    else:
        # Single layer prediction
        result = inference.predict_emotional_state(args.text, args.layer)
        
        print("\n" + "="*80)
        print("EMOTIONAL STATE PREDICTION")
        print("="*80)
        print(f"Text: '{result['text']}'")
        print(f"Layer: {result['layer']}")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nAll Probabilities:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")


if __name__ == "__main__":
    main()
