# Crisis Alert Probe

A system for training and using probes to detect emotional states in text using LLaMA-2 activations.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to LLaMA-2 models (requires Hugging Face authentication).

## Training Probes

### 1. Create Dataset
First, create a dataset from your CSV file containing text and emotional labels:

```bash
python src/create_dataset.py
```

Modify the script to point to your CSV file and desired output path.

### 2. Train Probes
Train probes across all layers (0-39) of LLaMA-2:

```bash
python src/train_reading_probe.py
```

This will:
- Load your dataset
- Split into train/validation sets (80/20)
- Train probes for each layer (0-39)
- Save best checkpoints in `probe_checkpoints/reading_probe/`
- Generate training metrics and reports

## Inference

### Command Line Interface
Run inference on new text:

```bash
# Single prediction using best layer
python src/inference_with_probes.py --text "Your text here"

# Specify layer
python src/inference_with_probes.py --text "Your text here" --layer 39

# Compare across top 5 layers
python src/inference_with_probes.py --text "Your text here" --compare
```

### Simple Example
For a minimal example, see:

```bash
python src/simple_inference_example.py
```

### Programmatic Usage
```python
from src.inference_with_probes import EmotionalStateInference

# Initialize
inference = EmotionalStateInference()

# Single prediction
result = inference.predict_emotional_state("I'm feeling great today!")
print(f"Predicted: {result['predicted_emotion']}")
print(f"Confidence: {result['confidence']:.4f}")

# Compare layers
results = inference.compare_layers("I'm feeling great today!", top_k=3)
```

## Output Format

The system predicts 7 emotional states:
- `very_happy`, `happy`, `slightly_positive`
- `neutral`
- `slightly_negative`, `sad`, `very_sad`

Each prediction includes:
- Predicted emotion
- Confidence score
- Probabilities for all emotions
- Layer used for prediction

## Files

- `src/train_reading_probe.py` - Main training script
- `src/inference_with_probes.py` - Full inference system
- `src/simple_inference_example.py` - Minimal inference example
- `src/create_dataset.py` - Dataset creation from CSV
- `probe_checkpoints/reading_probe/` - Trained probe checkpoints
