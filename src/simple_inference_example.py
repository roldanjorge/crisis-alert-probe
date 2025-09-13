#!/usr/bin/env python3
"""
Simple example showing how to use trained probes for inference.

This is a minimal example that demonstrates the core concepts:
1. Loading a trained probe
2. Extracting activations from LLaMA-2
3. Running inference
"""

import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from probes import Probe
from utils import llama_v2_prompt


def simple_inference_example():
    """Simple example of using a trained probe for inference."""
    
    # 1. Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # 2. Load LLaMA-2 model
    print("Loading LLaMA-2 model...")
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf",
        device=device,
        torch_dtype=torch.float16,
    )
    
    # 3. Load trained probe (using layer 39 - the best performing one)
    layer = 39
    print(f"Loading probe for layer {layer}...")
    
    # Define label mapping
    label2id = {
        "very_happy": 0,
        "happy": 1,
        "slightly_positive": 2,
        "neutral": 3,
        "slightly_negative": 4,
        "sad": 5,
        "very_sad": 6,
    }
    id2label = {v: k for k, v in label2id.items()}
    
    # Create and load probe
    probe = Probe(num_classes=len(label2id), device=device)
    probe.load_state_dict(torch.load(f"probe_checkpoints/reading_probe/probe_at_layer_{layer}.pth", 
                                    map_location=device))
    probe.eval()
    
    # 4. Prepare text for inference
    # text = "I'm so excited about my new job! Everything is going great!"
    text = "will anyone miss me if I die?"
    print(f"\nAnalyzing text: '{text}'")
    
    # Format text the same way as during training
    messages = [{"content": text, "role": "user"}]
    formatted_text = llama_v2_prompt(messages)
    formatted_text += " I think the emotional state of this user is"
    
    # 5. Extract activations
    print("Extracting activations...")
    with torch.no_grad():
        tokens = model.to_tokens(formatted_text)
        
        # Ensure sequence length constraint
        if tokens.shape[-1] > 2048:
            tokens = tokens[:, -2048:]
        
        # Run model with cache
        _, cache = model.run_with_cache(tokens.to(device), remove_batch_dim=False)
        
        # Get residual activation for the specified layer (last token)
        resid_post = cache["resid_post", layer][:, -1].detach().cpu().to(torch.float)
    
    # 6. Run inference
    print("Running inference...")
    with torch.no_grad():
        resid_post = resid_post.to(device)
        output = probe(resid_post)
        probabilities = output[0].cpu().numpy()[0]
        predicted_class = torch.argmax(output[0], dim=1).cpu().item()
    
    # 7. Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"Text: '{text}'")
    print(f"Layer: {layer}")
    print(f"Predicted Emotion: {id2label[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class]:.4f}")
    print(f"\nAll Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {id2label[i]}: {prob:.4f}")


if __name__ == "__main__":
    simple_inference_example()
