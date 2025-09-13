# Reading Probe Experimental Setup: Comprehensive Analysis

## Overview

This document provides a detailed analysis of the experimental setup for testing reading probes on emotional state classification using LLaMA-2-13b-chat-hf. The experiment evaluates the performance of trained probes that extract emotional information from transformer activations.

## 1. Experimental Architecture

### 1.1 Core Components

The experimental setup consists of three main components:

1. **LLaMA-2-13b-chat-hf Model**: A large language model serving as the base transformer
2. **Reading Probes**: Linear classifiers trained to extract emotional states from model activations
3. **Test Dataset**: Curated emotional state data for evaluation

### 1.2 Model Specifications

**LLaMA-2-13b-chat-hf Configuration:**
- **Parameters**: 12.7 billion parameters
- **Architecture**: Transformer with 40 layers, 40 attention heads
- **Hidden Dimension**: 5,120 (d_model)
- **Vocabulary**: 32,000 tokens
- **Context Length**: 4,096 tokens
- **Activation Function**: SiLU (Swish)
- **Normalization**: RMS Pre-normalization

## 2. Reading Probe Architecture

### 2.1 Probe Design

The reading probes are simple linear classifiers with the following architecture:

```python
class Probe(nn.Module):
    def __init__(self, num_classes=7, device="cuda"):
        self.input_dim = 5120  # LLaMA-2 hidden dimension
        self.num_classes = 7   # Emotional state categories
        self.classifier = nn.Sequential(
            nn.Linear(5120, 7), 
            nn.Sigmoid()
        )
```

### 2.2 Training Configuration

- **Learning Rate**: 1e-3
- **Weight Decay**: 0.1
- **Optimizer**: Adam with β₁=0.9, β₂=0.95
- **Regularization**: Weight decay applied to linear layers only
- **Scheduler**: ReduceLROnPlateau with factor=0.75

### 2.3 Probe Selection

**Layer 39 Probe**: Selected as the best-performing probe based on validation accuracy
- Achieved perfect accuracy (1.0) on validation set during training
- Located in the final layers of the transformer (layer 39/40)
- Captures high-level semantic representations before final output

## 3. Dataset and Data Processing

### 3.1 Test Dataset

**Current Dataset**: `genai_test_dataset_final_cleaned.csv`
- **Size**: 26,211 test cases
- **Format**: CSV with columns: `case_id`, `message`, `mood`
- **Emotional Categories**: 7 classes representing emotional valence spectrum

### 3.2 Emotional State Categories

The probe classifies text into 7 emotional states:

| ID | Label | Description |
|----|-------|-------------|
| 0 | very_happy | Extremely positive emotional state |
| 1 | happy | Positive emotional state |
| 2 | slightly_positive | Mildly positive emotional state |
| 3 | neutral | Neutral emotional state |
| 4 | slightly_negative | Mildly negative emotional state |
| 5 | sad | Negative emotional state |
| 6 | very_sad | Extremely negative emotional state |

### 3.3 Text Preprocessing

**Prompt Engineering**: Each test case is formatted using LLaMA-2's chat template:

```
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant...
<</SYS>>

{user_message} [/INST] I think the emotional state of this user is
```

**Key Features**:
- Consistent prompt structure matching training data
- System prompt for context
- Emotional state completion task
- Sequence length constraint (max 2048 tokens)

## 4. Experimental Procedure

### 4.1 Activation Extraction Process

For each test case, the following steps are performed:

1. **Tokenization**: Convert text to token IDs using LLaMA-2 tokenizer
2. **Sequence Truncation**: Limit to 2048 tokens if necessary
3. **Forward Pass**: Run model with caching enabled
4. **Activation Extraction**: Extract residual post-activations from layer 39
5. **Position Selection**: Use activations from the last token position

### 4.2 Probe Inference

```python
# Extract activations
_, cache = model.run_with_cache(tokens, remove_batch_dim=False)
resid_post = cache["resid_post", layer_39][:, -1]  # Last token

# Run probe inference
with torch.no_grad():
    output = probe(resid_post)
    probabilities = output[0].cpu().numpy()[0]
    predicted_class = torch.argmax(output[0], dim=1).cpu().item()
```

### 4.3 Memory Management

- **GPU Memory**: Automatic cleanup after each inference
- **Cache Clearing**: `torch.cuda.empty_cache()` after processing
- **Batch Processing**: Single sample processing to manage memory constraints

## 5. Evaluation Metrics

### 5.1 Primary Metrics

**Overall Accuracy**: 
- Formula: `Correct Predictions / Total Predictions`
- Interpretation: Percentage of correctly classified emotional states
- Range: [0, 1], higher is better

**Confusion Matrix**:
- Visual representation of prediction patterns
- Shows true vs predicted class distributions
- Identifies systematic misclassification patterns

### 5.2 Per-Class Metrics

**Precision**: 
- Formula: `True Positives / (True Positives + False Positives)`
- Interpretation: Accuracy of positive predictions for each class
- Measures: "When the model predicts class X, how often is it correct?"

**Recall**:
- Formula: `True Positives / (True Positives + False Negatives)`
- Interpretation: Coverage of actual positive cases
- Measures: "Of all actual class X cases, how many did the model find?"

**F1-Score**:
- Formula: `2 × (Precision × Recall) / (Precision + Recall)`
- Interpretation: Harmonic mean of precision and recall
- Balances precision and recall for imbalanced datasets

### 5.3 Weighted Metrics

**Weighted Average**: Metrics calculated by weighting each class by its support (number of true instances)
- Accounts for class imbalance in the dataset
- Provides overall performance assessment
- More robust than macro-averaging for imbalanced data

### 5.4 Confidence Analysis

**Prediction Confidence**:
- Maximum probability from the probe's output
- Range: [0, 1], higher indicates more confident predictions
- Used to analyze model calibration and uncertainty

**Confidence-Accuracy Relationship**:
- Correlation between prediction confidence and correctness
- Indicates model calibration quality
- Higher correlation suggests better uncertainty estimation

## 6. Visualization and Analysis

### 6.1 Generated Visualizations

1. **Confusion Matrix Heatmap**
   - Shows prediction patterns across all classes
   - Identifies systematic biases and errors
   - Color-coded for easy interpretation

2. **Per-Class Accuracy Bar Chart**
   - Individual performance for each emotional state
   - Reveals class-specific strengths and weaknesses
   - Helps identify challenging emotional categories

3. **Confidence Distribution Histograms**
   - Overall confidence distribution
   - Confidence by prediction correctness
   - Reveals model calibration patterns

4. **Confidence vs Accuracy Scatter Plot**
   - Relationship between confidence and correctness
   - Trend line showing calibration quality
   - Identifies overconfident or underconfident predictions

5. **Per-Class Performance Metrics**
   - Precision, recall, and F1-score for each class
   - Side-by-side comparison of metrics
   - Identifies trade-offs between precision and recall

6. **Error Analysis Chart**
   - Most common misclassification patterns
   - Top 10 error types with frequency
   - Helps understand model limitations

### 6.2 Dynamic Class Handling

The evaluation system automatically adapts to datasets with different class distributions:
- **Dynamic Detection**: Identifies classes present in the data
- **Adaptive Metrics**: Computes metrics only for present classes
- **Flexible Visualizations**: Adjusts plots based on available classes

## 7. Technical Implementation

### 7.1 Software Stack

- **PyTorch**: Deep learning framework for model operations
- **TransformerLens**: Library for transformer interpretability
- **scikit-learn**: Machine learning metrics and evaluation
- **Matplotlib/Seaborn**: Visualization and plotting
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations

### 7.2 Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for inference speed
- **Memory**: Sufficient VRAM for LLaMA-2-13b model (~24GB recommended)
- **Storage**: Space for model weights and checkpoint files

### 7.3 Performance Considerations

- **Inference Speed**: ~2.8 samples/second on CUDA GPU
- **Memory Usage**: Automatic cleanup prevents memory leaks
- **Batch Processing**: Single-sample processing for memory efficiency
- **Caching**: Model activations cached during forward pass

## 8. Experimental Results Interpretation

### 8.1 Expected Outcomes

**High Performance Indicators**:
- Overall accuracy > 0.8
- Balanced precision/recall across classes
- Strong confidence-accuracy correlation
- Minimal systematic bias in confusion matrix

**Performance Challenges**:
- Class imbalance effects
- Similar emotional states confusion (e.g., sad vs very_sad)
- Context-dependent emotional interpretation
- Ambiguous or neutral emotional content

### 8.2 Error Analysis

**Common Error Patterns**:
- Adjacent emotional states confusion
- Context-dependent misclassification
- Ambiguous emotional expressions
- Cultural or linguistic variations

**Model Limitations**:
- Binary classification bias
- Limited emotional nuance capture
- Context sensitivity issues
- Generalization across domains

## 9. Reproducibility and Validation

### 9.1 Reproducibility Features

- **Deterministic Processing**: Consistent tokenization and formatting
- **Version Control**: Tracked model and probe versions
- **Documentation**: Comprehensive setup documentation
- **Modular Design**: Separable components for easy replication

### 9.2 Validation Methodology

- **Cross-Dataset Testing**: Multiple dataset validation
- **Statistical Significance**: Large sample sizes for reliable metrics
- **Error Analysis**: Detailed misclassification examination
- **Confidence Calibration**: Uncertainty estimation validation

## 10. Future Directions

### 10.1 Potential Improvements

- **Multi-Layer Analysis**: Testing probes across different transformer layers
- **Ensemble Methods**: Combining multiple probe predictions
- **Fine-tuning**: Domain-specific probe adaptation
- **Interpretability**: Understanding probe decision mechanisms

### 10.2 Research Applications

- **Emotional AI**: Applications in mental health and therapy
- **Content Moderation**: Automated emotional content analysis
- **Human-Computer Interaction**: Emotion-aware interfaces
- **Psychological Research**: Large-scale emotional pattern analysis

---

## Conclusion

This experimental setup provides a comprehensive framework for evaluating reading probes on emotional state classification. The combination of LLaMA-2's powerful representations, carefully designed probes, and thorough evaluation metrics offers insights into how transformer models encode and can be probed for emotional understanding. The dynamic evaluation system ensures robust testing across diverse datasets and emotional categories, making it a valuable tool for mechanistic interpretability research.
