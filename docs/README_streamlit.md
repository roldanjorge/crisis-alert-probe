# 🧠 LLaMA-2 Chat App with Emotional Monitoring

A Streamlit application that provides a ChatGPT-like interface with LLaMA-2 and real-time emotional state monitoring using your trained probes.

## 🌟 Features

### 💬 Chat Interface
- **ChatGPT-like experience** with LLaMA-2-13b-chat-hf
- **Conversation history** maintained throughout the session
- **Real-time responses** with proper formatting
- **Context awareness** using conversation history

### 📊 Emotional State Monitoring
- **Real-time analysis** of user emotional state using trained probes
- **Visual probability distributions** for all 7 emotional states
- **Confidence tracking** over time
- **Emotion statistics** showing frequency of different states
- **Timeline visualization** of emotional changes

### 🎨 User Interface
- **Two-column layout**: Emotional monitoring on left, chat on right
- **Interactive visualizations** using Plotly
- **Real-time updates** as you chat
- **Export functionality** for chat data and emotional analysis
- **Responsive design** that works on different screen sizes

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Trained probes in `probe_checkpoints/reading_probe/`
- Sufficient GPU memory for LLaMA-2-13b (or use CPU)

### Installation

1. **Install requirements:**
```bash
pip install -r ../requirements.txt
```

2. **Run the app:**
```bash
# Option 1: Use the launcher script
python run_streamlit_app.py

# Option 2: Run directly with Streamlit
streamlit run streamlit_chat_app.py
```

3. **Open your browser** to `http://localhost:8501`

## 📱 App Interface

### Left Column - Emotional Monitoring
- **Current Emotional State**: Shows the latest predicted emotion with confidence
- **Probability Visualization**: Bar chart showing all emotional state probabilities
- **Timeline Chart**: Line graph showing confidence changes over time
- **Emotion Statistics**: Frequency breakdown of emotions in the conversation
- **Model Information**: Details about the probe and model being used

### Right Column - Chat Interface
- **Chat Messages**: Conversation history with user and assistant messages
- **Input Field**: Type your messages here
- **Real-time Analysis**: Each message is analyzed for emotional state
- **LLaMA-2 Responses**: Generated responses using your trained model

### Sidebar Controls
- **Clear Chat**: Reset the conversation and emotional history
- **Export Data**: Download chat and emotional analysis as JSON
- **Model Settings**: View probe layer and accuracy information
- **Available Layers**: See performance of different probe layers

## 🔧 Technical Details

### Model Architecture
- **Base Model**: LLaMA-2-13b-chat-hf via transformer_lens
- **Probe Layer**: Layer 39 (perfect accuracy: 1.0)
- **Emotional States**: 7 categories (very_happy to very_sad)
- **Input Format**: Same as training (llama_v2_prompt + emotional state prompt)

### Performance
- **Probe Accuracy**: Layer 39 achieved 100% accuracy on validation set
- **Response Generation**: Uses temperature=0.7 for natural responses
- **Context Length**: Maintains last 10 messages for context
- **Memory Management**: Efficient caching and GPU memory usage

### Data Flow
1. **User Input** → Text message
2. **Emotional Analysis** → Probe extracts emotional state
3. **Response Generation** → LLaMA-2 generates response
4. **Visualization Update** → Charts and statistics update
5. **Session Storage** → Data saved in Streamlit session state

## 📊 Emotional States

The app monitors 7 emotional states:

| State | Description | Example |
|-------|-------------|---------|
| very_happy | Extremely positive | "I'm over the moon!" |
| happy | Generally positive | "I'm doing great!" |
| slightly_positive | Mildly positive | "Things are okay" |
| neutral | Neither positive nor negative | "Just normal" |
| slightly_negative | Mildly negative | "A bit disappointed" |
| sad | Generally negative | "I'm feeling down" |
| very_sad | Extremely negative | "I'm devastated" |

## 🎯 Use Cases

### Personal Use
- **Emotional journaling** with AI companion
- **Mood tracking** over conversations
- **Self-awareness** through emotional analysis
- **Therapeutic conversations** with emotional monitoring

### Research Applications
- **Emotional state analysis** in human-AI interactions
- **Conversation dynamics** research
- **Probe performance** evaluation in real-time
- **User behavior** analysis

### Educational Purposes
- **Understanding emotional AI** and probe technology
- **Learning about transformer** architectures
- **Exploring mechanistic interpretability** concepts
- **Demonstrating AI capabilities** to students

## 🔍 Advanced Features

### Multi-Layer Analysis
The app can be extended to compare emotional predictions across multiple layers:

```python
# In the app, you can modify the probe loading to use different layers
best_layers = [39, 34, 30, 26, 33]  # Top 5 performing layers
```

### Custom Prompts
You can modify the emotional analysis prompt in the `extract_emotional_state` function:

```python
# Current prompt
formatted_text += " I think the emotional state of this user is"

# Custom prompt example
formatted_text += " Analyze the emotional tone of this message:"
```

### Export and Analysis
The app exports data in JSON format including:
- Complete conversation history
- Timestamped emotional analyses
- Probability distributions for each message
- Confidence scores over time

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size or use CPU
   - Close other GPU applications
   - Use smaller model if available

2. **Probe Not Found**
   - Check `probe_checkpoints/reading_probe/` exists
   - Verify `probe_training_metrics.csv` is present
   - Ensure you're running from the correct directory

3. **Slow Loading**
   - First run takes longer due to model downloading
   - Subsequent runs use cached model
   - Consider using smaller model for faster loading

4. **Import Errors**
   - Install all requirements: `pip install -r ../requirements.txt`
   - Check Python version (3.8+ required)
   - Verify transformer_lens installation

### Performance Optimization

1. **Memory Usage**
   - Use `torch_dtype=torch.float16` for model
   - Clear cache between operations
   - Limit conversation history length

2. **Speed Improvements**
   - Cache model and probe loading
   - Use GPU acceleration
   - Optimize sequence length

## 🔮 Future Enhancements

### Planned Features
- **Multi-model support** (different LLaMA sizes)
- **Custom probe training** interface
- **Advanced visualizations** (heatmaps, clustering)
- **Emotional trend analysis** with predictions
- **Multi-user support** with separate sessions
- **API integration** for external applications

### Extensibility
The app is designed to be easily extensible:
- **New emotional states** can be added by retraining probes
- **Different models** can be integrated
- **Custom visualizations** can be added
- **Additional analysis** can be incorporated

## 📝 Example Usage

1. **Start the app**: `python run_streamlit_app.py`
2. **Type a message**: "I'm feeling really excited about my new project!"
3. **See emotional analysis**: App shows "happy" with high confidence
4. **Get LLaMA-2 response**: AI responds appropriately to your excitement
5. **Continue chatting**: Each message updates the emotional monitoring
6. **Export data**: Download your conversation and emotional analysis

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of your mechanistic interpretability exploration. Feel free to modify and extend as needed for your research and applications.

---

**Enjoy exploring emotional AI with your trained probes! 🎉**
