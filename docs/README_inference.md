# Using Trained Probes for Emotional State Inference

This guide explains how to use your trained emotional state probes with LLaMA-2 for inference on new text.

## Overview

You have successfully trained probes on 40 layers of LLaMA-2-13b-chat-hf to classify emotional states into 7 categories:

- **very_happy** (0)
- **happy** (1) 
- **slightly_positive** (2)
- **neutral** (3)
- **slightly_negative** (4)
- **sad** (5)
- **very_sad** (6)

## Performance Summary

Based on your training metrics, here are the top-performing layers:

| Layer | Best Accuracy |
|-------|---------------|
| 39    | 1.000000      |
| 34    | 0.994975      |
| 30    | 0.994975      |
| 26    | 0.994975      |
| 33    | 0.994975      |

**Layer 39 achieved perfect accuracy (1.0) on your validation set!**

## Quick Start

### 1. Simple Inference Example

```python
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from probes import Probe
from utils import llama_v2_prompt

# Setup
device = get_device()
model = HookedTransformer.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf",
    device=device,
    torch_dtype=torch.float16,
)

# Load trained probe (layer 39 - best performing)
layer = 39
probe = Probe(num_classes=7, device=device)
probe.load_state_dict(torch.load(f"probe_checkpoints/reading_probe/probe_at_layer_{layer}.pth", 
                                map_location=device))
probe.eval()

# Define label mapping
id2label = {0: 'very_happy', 1: 'happy', 2: 'slightly_positive', 
           3: 'neutral', 4: 'slightly_negative', 5: 'sad', 6: 'very_sad'}

# Analyze text
text = "I'm so excited about my new job! Everything is going great!"

# Format text (same as training)
messages = [{"content": text, "role": "user"}]
formatted_text = llama_v2_prompt(messages)
formatted_text += " I think the emotional state of this user is"

# Extract activations
with torch.no_grad():
    tokens = model.to_tokens(formatted_text)
    if tokens.shape[-1] > 2048:
        tokens = tokens[:, -2048:]
    
    _, cache = model.run_with_cache(tokens.to(device), remove_batch_dim=False)
    resid_post = cache["resid_post", layer][:, -1].detach().cpu().to(torch.float)

# Run inference
with torch.no_grad():
    resid_post = resid_post.to(device)
    output = probe(resid_post)
    probabilities = output[0].cpu().numpy()[0]
    predicted_class = torch.argmax(output[0], dim=1).cpu().item()

# Display results
print(f"Text: '{text}'")
print(f"Predicted Emotion: {id2label[predicted_class]}")
print(f"Confidence: {probabilities[predicted_class]:.4f}")
```

### 2. Using the Comprehensive Inference Script

```bash
# Single layer prediction
python inference_with_probes.py --text "I'm feeling great today!" --layer 39

# Compare across top 5 layers
python inference_with_probes.py --text "I'm having mixed feelings about this." --compare

# Use different model
python inference_with_probes.py --text "This is amazing!" --model "meta-llama/Llama-2-7b-chat-hf"
```

### 3. Using the Simple Example Script

```bash
python simple_inference_example.py
```

## Key Concepts

### 1. Text Formatting
Always use the same text formatting as during training:
- Use `llama_v2_prompt()` to format messages
- Append " I think the emotional state of this user is" to the text
- This ensures consistency with your training data

### 2. Activation Extraction
- Use `model.run_with_cache()` to get activations
- Extract `resid_post` from the specified layer
- Use the last token position (`[:, -1]`)
- Apply the same sequence length constraint (max 2048 tokens)

### 3. Probe Loading
- Load probe weights using `torch.load()`
- Try best checkpoint first (`probe_at_layer_X.pth`), then final checkpoint (`probe_at_layer_X_final.pth`)
- Set probe to eval mode with `probe.eval()`

### 4. Layer Selection
- **Layer 39**: Perfect accuracy (1.0) - recommended for most use cases
- **Layers 26, 30, 33, 34, 35, 37**: Very high accuracy (>99%) - good alternatives
- **Earlier layers (0-25)**: Lower accuracy - use only if you need early-layer representations

## Advanced Usage

### Comparing Multiple Layers

```python
# Load training metrics
import pandas as pd
metrics_df = pd.read_csv("probe_checkpoints/reading_probe/probe_training_metrics.csv")
best_layers = metrics_df.groupby('layer')['best_accuracy'].max().sort_values(ascending=False)

# Compare top 5 layers
top_layers = best_layers.head(5).index.tolist()
for layer in top_layers:
    result = predict_emotional_state(text, layer)
    print(f"Layer {layer}: {result['predicted_emotion']} (confidence: {result['confidence']:.4f})")
```

### Ensemble Predictions

```python
# Average predictions across top layers
top_layers = [39, 34, 30, 26, 33]  # Top 5 layers
all_probs = []

for layer in top_layers:
    probe = load_probe(layer)
    # ... extract activations and run inference ...
    all_probs.append(probabilities)

# Average probabilities
ensemble_probs = np.mean(all_probs, axis=0)
ensemble_prediction = np.argmax(ensemble_probs)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use `torch_dtype=torch.float16` when loading the model
   - Reduce batch size or sequence length
   - Use CPU if necessary: `device="cpu"`

2. **Checkpoint Not Found**
   - Check that checkpoint files exist in `probe_checkpoints/reading_probe/`
   - Verify layer number is between 0-39
   - Try both `.pth` and `_final.pth` files

3. **Inconsistent Predictions**
   - Ensure text formatting matches training (use `llama_v2_prompt`)
   - Check that you're using the last token position (`[:, -1]`)
   - Verify sequence length constraints

### Performance Tips

1. **Use the best layer**: Layer 39 achieved perfect accuracy
2. **Cache activations**: Extract activations once and reuse for multiple probes
3. **Batch processing**: Process multiple texts together when possible
4. **Memory management**: Clear cache between batches with `torch.cuda.empty_cache()`

## Files Created

- `inference_with_probes.py`: Comprehensive inference script with command-line interface
- `simple_inference_example.py`: Minimal example showing core concepts
- `probe_inference_tutorial.ipynb`: Interactive Jupyter notebook (if created)
- `README_inference.md`: This guide

## Next Steps

1. **Try the examples**: Run the simple example script first
2. **Experiment with different texts**: Test various emotional expressions
3. **Compare layers**: See how predictions vary across layers
4. **Build applications**: Integrate into your own projects
5. **Analyze confidence**: Use probability distributions to understand prediction reliability

Your trained probes are ready to use! Layer 39 with perfect accuracy should give you excellent results for emotional state classification.
