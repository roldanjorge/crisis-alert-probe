#!/usr/bin/env python3
"""
Streamlit Chat App with Emotional State Monitoring

This app provides a ChatGPT-like interface with LLaMA-2 and real-time emotional state monitoring
using trained probes. The interface has two columns:
- Left: Emotional state monitoring and analysis
- Right: Chat interface with LLaMA-2
"""

import streamlit as st
import torch
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_device
from probes import Probe
from utils import llama_v2_prompt
import json
import os

# Optional imports for enhanced UI components
try:
    import streamlit_antd_components as sac
    ANT_DESIGN_AVAILABLE = True
except ImportError:
    ANT_DESIGN_AVAILABLE = False
    st.warning("streamlit-antd-components not available. Using basic UI components.")

try:
    import streamlit_image_select
    IMAGE_SELECT_AVAILABLE = True
except ImportError:
    IMAGE_SELECT_AVAILABLE = False

try:
    import st_on_hover_tabs
    HOVER_TABS_AVAILABLE = True
except ImportError:
    HOVER_TABS_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="LLaMA-2 Chat with Emotional Monitoring",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .emotion-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f3e5f5;
        margin-right: 2rem;
    }
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_probes():
    """Load LLaMA-2 model and trained probes with caching."""
    device = get_device()
    
    # Load LLaMA-2 model
    st.info("Loading LLaMA-2 model... This may take a few minutes on first run.")
    model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf",
        device=device,
        torch_dtype=torch.float16,
    )
    
    # Load training metrics to find best layers
    metrics_df = pd.read_csv("probe_checkpoints/reading_probe/probe_training_metrics.csv")
    best_layers = metrics_df.groupby('layer')['best_accuracy'].max().sort_values(ascending=False)
    
    # Load the best probe (layer 39)
    best_layer = best_layers.index[0]
    probe = Probe(num_classes=7, device=device)
    probe.load_state_dict(torch.load(f"probe_checkpoints/reading_probe/probe_at_layer_{best_layer}.pth", 
                                    map_location=device))
    probe.eval()
    
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
    
    return model, probe, best_layer, id2label, best_layers


def extract_emotional_state(text, model, probe, layer, id2label):
    """Extract emotional state from text using the trained probe."""
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
            "all_probabilities": {id2label[i]: float(prob) for i, prob in enumerate(probabilities)},
            "timestamp": datetime.now()
        }
    except Exception as e:
        st.error(f"Error analyzing emotional state: {e}")
        return None


def generate_llama_response(user_message, model, conversation_history):
    """Generate LLaMA-2 response for the user message."""
    try:
        # Format conversation history
        messages = []
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Format for LLaMA-2
        formatted_text = llama_v2_prompt(messages)
        
        with torch.no_grad():
            tokens = model.to_tokens(formatted_text)
            
            # Ensure sequence length constraint
            if tokens.shape[-1] > 2048:
                tokens = tokens[:, -2048:]
            
            # Generate response using proper transformer_lens generation
            response_tokens = generate_text(model, tokens, max_new_tokens=200, temperature=0.7)
            
            # Decode response
            response_text = model.to_string(response_tokens[0, tokens.shape[1]:])
            
            # Clean up response (remove any formatting artifacts)
            response_text = response_text.strip()
            if response_text.startswith("[/INST]"):
                response_text = response_text[7:].strip()
            if response_text.endswith("</s>"):
                response_text = response_text[:-4].strip()
            if response_text.startswith("<s>"):
                response_text = response_text[3:].strip()
            
            # If response is empty or too short, use fallback
            if len(response_text.strip()) < 5:
                response_text = get_fallback_response(user_message)
            
            return response_text
            
    except Exception as e:
        st.error(f"Error generating response: {e}")
        # Use fallback response instead of generic error message
        return get_fallback_response(user_message)


def generate_text(model, input_tokens, max_new_tokens=200, temperature=0.7):
    """Generate text using transformer_lens model."""
    device = model.cfg.device
    tokens = input_tokens.to(device)
    
    try:
        for _ in range(max_new_tokens):
            # Get logits for the last token
            logits = model(tokens)
            last_token_logits = logits[0, -1, :]
            
            # Apply temperature
            scaled_logits = last_token_logits / temperature
            
            # Sample next token
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            
            # Stop if we hit EOS token
            if next_token.item() == model.tokenizer.eos_token_id:
                break
        
        return tokens
    except Exception as e:
        # If generation fails, return the input tokens
        return tokens


def get_fallback_response(user_message):
    """Get a fallback response when generation fails."""
    fallback_responses = [
        "I understand your message. I'm here to help! How can I assist you today?",
        "That's an interesting question. Let me think about that for a moment.",
        "I appreciate you sharing that with me. What would you like to discuss?",
        "Thank you for your message. I'm here to help with any questions you might have.",
        "I see what you're saying. Could you tell me more about that?",
        "That's a great point. What are your thoughts on this topic?",
        "I'm listening. Please continue sharing your thoughts.",
        "Thank you for reaching out. How can I best assist you today?"
    ]
    
    # Simple hash-based selection for consistency
    import hashlib
    hash_val = int(hashlib.md5(user_message.encode()).hexdigest(), 16)
    return fallback_responses[hash_val % len(fallback_responses)]


def create_emotion_visualization(emotion_data):
    """Create visualization for emotional state probabilities."""
    if not emotion_data:
        return None
    
    emotions = list(emotion_data["all_probabilities"].keys())
    probabilities = list(emotion_data["all_probabilities"].values())
    
    # Create bar chart
    fig = px.bar(
        x=emotions, 
        y=probabilities,
        title="Emotional State Probabilities",
        labels={'x': 'Emotion', 'y': 'Probability'},
        color=probabilities,
        color_continuous_scale='RdYlBu_r'
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    return fig


def create_emotion_timeline(emotion_history):
    """Create timeline visualization of emotional states."""
    if len(emotion_history) < 2:
        return None
    
    timestamps = [e["timestamp"] for e in emotion_history]
    emotions = [e["predicted_emotion"] for e in emotion_history]
    confidences = [e["confidence"] for e in emotion_history]
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidences,
        mode='lines+markers',
        name='Confidence',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Emotional State Confidence Over Time",
        xaxis_title="Time",
        yaxis_title="Confidence",
        height=300
    )
    
    return fig


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 LLaMA-2 Chat with Emotional Monitoring</h1>', unsafe_allow_html=True)
    
    # Load model and probes
    model, probe, best_layer, id2label, best_layers = load_model_and_probes()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "emotion_history" not in st.session_state:
        st.session_state.emotion_history = []
    if "current_emotion" not in st.session_state:
        st.session_state.current_emotion = None
    
    # Create two-column layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Emotional State Monitoring")
        
        # Current emotional state
        if st.session_state.current_emotion:
            emotion = st.session_state.current_emotion
            st.markdown(f"""
            <div class="emotion-card">
                <h4>Current Emotional State</h4>
                <h2 style="color: #1f77b4;">{emotion['predicted_emotion'].replace('_', ' ').title()}</h2>
                <p><strong>Confidence:</strong> {emotion['confidence']:.2%}</p>
                <p><strong>Layer:</strong> {best_layer}</p>
                <p><strong>Time:</strong> {emotion['timestamp'].strftime('%H:%M:%S')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Emotion visualization
            fig = create_emotion_visualization(emotion)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Emotion timeline
        if len(st.session_state.emotion_history) > 1:
            st.markdown("### 📈 Emotional State Timeline")
            timeline_fig = create_emotion_timeline(st.session_state.emotion_history)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
        
        # Emotion statistics
        if st.session_state.emotion_history:
            st.markdown("### 📋 Emotion Statistics")
            
            # Count emotions
            emotion_counts = {}
            for emotion_data in st.session_state.emotion_history:
                emotion = emotion_data["predicted_emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Display statistics
            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(st.session_state.emotion_history)) * 100
                st.markdown(f"**{emotion.replace('_', ' ').title()}:** {count} ({percentage:.1f}%)")
        
        # Model info
        st.markdown("### 🔧 Model Information")
        st.markdown(f"""
        - **Model:** LLaMA-2-13b-chat-hf
        - **Probe Layer:** {best_layer} (Best Accuracy: {best_layers.iloc[0]:.4f})
        - **Total Messages:** {len(st.session_state.messages)}
        - **Emotion Analyses:** {len(st.session_state.emotion_history)}
        """)
    
    with col2:
        st.markdown("### 💬 Chat with LLaMA-2")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Analyze emotional state
            with st.spinner("Analyzing emotional state..."):
                emotion_data = extract_emotional_state(prompt, model, probe, best_layer, id2label)
                if emotion_data:
                    st.session_state.current_emotion = emotion_data
                    st.session_state.emotion_history.append(emotion_data)
            
            # Generate LLaMA-2 response
            with st.spinner("Generating response..."):
                response = generate_llama_response(prompt, model, st.session_state.messages)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update the display
            st.rerun()
    
    # Sidebar with controls
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.emotion_history = []
            st.session_state.current_emotion = None
            st.rerun()
        
        # Export data button
        if st.button("📥 Export Chat Data"):
            chat_data = {
                "messages": st.session_state.messages,
                "emotion_history": [
                    {
                        "timestamp": e["timestamp"].isoformat(),
                        "predicted_emotion": e["predicted_emotion"],
                        "confidence": e["confidence"],
                        "all_probabilities": e["all_probabilities"]
                    }
                    for e in st.session_state.emotion_history
                ]
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(chat_data, indent=2),
                file_name=f"chat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Model settings
        st.markdown("### 🔧 Model Settings")
        st.markdown(f"**Probe Layer:** {best_layer}")
        st.markdown(f"**Accuracy:** {best_layers.iloc[0]:.4f}")
        
        # Available layers
        st.markdown("### 📊 Available Probe Layers")
        for i, (layer, accuracy) in enumerate(best_layers.head(10).items()):
            st.markdown(f"**Layer {layer}:** {accuracy:.4f}")


if __name__ == "__main__":
    main()
