#!/usr/bin/env python3
"""
Demo script showing emotional state monitoring with different text examples.

This script demonstrates how the emotional monitoring works with various text inputs
before running the full Streamlit app.
"""

import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from probes import Probe
from utils import llama_v2_prompt
import pandas as pd

def load_model_and_probe():
    """Load LLaMA-2 model and trained probe."""
    print("🔄 Loading LLaMA-2 model and probe...")
    
    device = get_device()
    
    # Load LLaMA-2 model
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf",
        device=device,
        torch_dtype=torch.float16,
    )
    
    # Load training metrics
    metrics_df = pd.read_csv("probe_checkpoints/reading_probe/probe_training_metrics.csv")
    best_layers = metrics_df.groupby('layer')['best_accuracy'].max().sort_values(ascending=False)
    best_layer = best_layers.index[0]
    
    # Load the best probe
    probe = Probe(num_classes=7, device=device)
    probe.load_state_dict(torch.load(f"probe_checkpoints/reading_probe/probe_at_layer_{best_layer}.pth", 
                                    map_location=device))
    probe.eval()
    
    # Define label mapping
    id2label = {0: 'very_happy', 1: 'happy', 2: 'slightly_positive', 
               3: 'neutral', 4: 'slightly_negative', 5: 'sad', 6: 'very_sad'}
    
    print(f"✅ Model loaded on {device}")
    print(f"✅ Probe loaded for layer {best_layer} (accuracy: {best_layers.iloc[0]:.4f})")
    
    return model, probe, best_layer, id2label

def analyze_emotional_state(text, model, probe, layer, id2label):
    """Analyze emotional state of text using the trained probe."""
    try:
        # Format text the same way as during training
        messages = [{"content": text, "role": "user"}]
        formatted_text = llama_v2_prompt(messages)
        formatted_text += " I think the emotional state of this user is"
        
        with torch.no_grad():
            tokens = model.to_tokens(formatted_text)
            
            # Ensure sequence length constraint
            if tokens.shape[-1] > 2048:
                tokens = tokens[:, -2048:]
            
            # Run model with cache
            _, cache = model.run_with_cache(tokens.to(model.cfg.device), remove_batch_dim=False)
            
            # Get residual activation for the specified layer (last token)
            resid_post = cache["resid_post", layer][:, -1].detach().cpu().to(torch.float)
            
            # Run inference
            resid_post = resid_post.to(probe.device)
            output = probe(resid_post)
            probabilities = output[0].cpu().numpy()[0]
            predicted_class = torch.argmax(output[0], dim=1).cpu().item()
        
        return {
            "predicted_emotion": id2label[predicted_class],
            "confidence": float(probabilities[predicted_class]),
            "all_probabilities": {id2label[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
    except Exception as e:
        print(f"❌ Error analyzing text: {e}")
        return None

def print_analysis(text, result):
    """Print emotional analysis results in a formatted way."""
    print(f"\n📝 Text: '{text}'")
    print(f"🎯 Predicted Emotion: {result['predicted_emotion'].replace('_', ' ').title()}")
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print("📈 All Probabilities:")
    
    # Sort probabilities by value
    sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs:
        bar_length = int(prob * 20)  # Scale to 20 characters
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  {emotion.replace('_', ' ').title():<20} {prob:.3f} {bar}")

def main():
    """Run emotional monitoring demo."""
    print("🧠 Emotional State Monitoring Demo")
    print("=" * 50)
    
    # Load model and probe
    model, probe, best_layer, id2label = load_model_and_probe()
    
    # Test texts with different emotional states
    test_texts = [
        "I'm so excited about my new job! Everything is going great!",
        "I've been feeling really down lately. Nothing seems to be working out.",
        "The weather is okay today. I'm just going about my usual routine.",
        "I'm having mixed feelings about this situation.",
        "This is absolutely terrible! I can't believe this happened!",
        "I'm feeling pretty good today, all things considered.",
        "will anyone miss me if I die?",
        "I just won the lottery! This is the best day of my life!",
        "I'm not sure how I feel about this. It's complicated.",
        "Everything is falling apart. I don't know what to do anymore."
    ]
    
    print(f"\n🔍 Analyzing {len(test_texts)} test texts...")
    print("=" * 50)
    
    results = []
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[{i}/{len(test_texts)}] Analyzing...")
        
        result = analyze_emotional_state(text, model, probe, best_layer, id2label)
        if result:
            print_analysis(text, result)
            results.append((text, result))
        else:
            print(f"❌ Failed to analyze: '{text}'")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    emotion_counts = {}
    total_confidence = 0
    
    for text, result in results:
        emotion = result['predicted_emotion']
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        total_confidence += result['confidence']
    
    print("🎭 Emotion Distribution:")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results)) * 100
        print(f"  {emotion.replace('_', ' ').title():<20} {count:2d} ({percentage:4.1f}%)")
    
    avg_confidence = total_confidence / len(results)
    print(f"\n📈 Average Confidence: {avg_confidence:.2%}")
    print(f"🎯 Layer Used: {best_layer}")
    print(f"✅ Successful Analyses: {len(results)}/{len(test_texts)}")
    
    print("\n🚀 Ready to run the Streamlit app!")
    print("Run: python run_streamlit_app.py")

if __name__ == "__main__":
    main()
