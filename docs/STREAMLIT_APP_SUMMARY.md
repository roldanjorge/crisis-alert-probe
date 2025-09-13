# 🧠 Streamlit Chat App with Emotional Monitoring - Complete Setup

## 🎯 What You Now Have

I've created a complete Streamlit application that provides a ChatGPT-like interface with LLaMA-2 and real-time emotional state monitoring using your trained probes.

### 📁 Files Created

1. **`streamlit_chat_app.py`** - Main Streamlit application
2. **`run_streamlit_app.py`** - Launcher script with error checking
3. **`test_streamlit_setup.py`** - Setup verification script
4. **`demo_emotional_monitoring.py`** - Demo of emotional monitoring
5. **`README_streamlit.md`** - Comprehensive documentation
6. **`STREAMLIT_APP_SUMMARY.md`** - This summary

**Note:** All required packages are now in the main `../requirements.txt` file

## 🚀 Quick Start Guide

### Step 1: Install Requirements
```bash
cd /teamspace/studios/this_studio/mech_interp_exploration/src
pip install -r ../requirements.txt
```

### Step 2: Test Your Setup
```bash
python test_streamlit_setup.py
```

### Step 3: Try the Demo
```bash
python demo_emotional_monitoring.py
```

### Step 4: Launch the App
```bash
python run_streamlit_app.py
```

### Step 4: Open Your Browser
The app will automatically open at `http://localhost:8501`

## 🎨 App Features

### Left Column - Emotional Monitoring
- **Real-time emotional analysis** of your messages
- **Visual probability distributions** for all 7 emotional states
- **Confidence tracking** over time
- **Emotion statistics** showing frequency breakdown
- **Timeline visualization** of emotional changes

### Right Column - Chat Interface
- **ChatGPT-like experience** with LLaMA-2-13b-chat-hf
- **Conversation history** maintained throughout session
- **Real-time responses** with proper formatting
- **Context awareness** using conversation history

### Sidebar Controls
- **Clear Chat** - Reset conversation and emotional history
- **Export Data** - Download chat and emotional analysis as JSON
- **Model Settings** - View probe layer and accuracy information
- **Available Layers** - See performance of different probe layers

## 🔧 Technical Implementation

### Model Architecture
- **Base Model**: LLaMA-2-13b-chat-hf via transformer_lens
- **Probe Layer**: Layer 39 (perfect accuracy: 1.0)
- **Emotional States**: 7 categories (very_happy to very_sad)
- **Input Format**: Same as training (llama_v2_prompt + emotional state prompt)

### Key Functions
- **`load_model_and_probes()`**: Cached loading of model and probe
- **`extract_emotional_state()`**: Real-time emotional analysis
- **`generate_llama_response()`**: LLaMA-2 response generation
- **`create_emotion_visualization()`**: Interactive probability charts
- **`create_emotion_timeline()`**: Confidence tracking over time

### Data Flow
1. **User Input** → Text message
2. **Emotional Analysis** → Probe extracts emotional state
3. **Response Generation** → LLaMA-2 generates response
4. **Visualization Update** → Charts and statistics update
5. **Session Storage** → Data saved in Streamlit session state

## 📊 Emotional States Monitored

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

## 🔍 Example Usage

1. **Start the app**: `python run_streamlit_app.py`
2. **Type a message**: "I'm feeling really excited about my new project!"
3. **See emotional analysis**: App shows "happy" with high confidence
4. **Get LLaMA-2 response**: AI responds appropriately to your excitement
5. **Continue chatting**: Each message updates the emotional monitoring
6. **Export data**: Download your conversation and emotional analysis

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
   - Install all requirements: `pip install -r requirements_streamlit.txt`
   - Check Python version (3.8+ required)
   - Verify transformer_lens installation

### Performance Tips

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

## 📝 Key Benefits

1. **Real-time Analysis**: Every message is analyzed for emotional state
2. **Visual Feedback**: Interactive charts show emotional probabilities
3. **Historical Tracking**: See how emotions change over time
4. **Export Capability**: Download data for further analysis
5. **User-friendly**: ChatGPT-like interface that's easy to use
6. **Research-ready**: Built for mechanistic interpretability research

## 🎉 Ready to Use!

Your Streamlit app is now ready to provide a ChatGPT-like experience with real-time emotional monitoring using your trained probes. The app combines:

- **Your trained emotional state probes** (Layer 39 with perfect accuracy)
- **LLaMA-2-13b-chat-hf** for natural conversation
- **Real-time emotional analysis** with visual feedback
- **Interactive visualizations** using Plotly
- **Session management** for conversation history
- **Export functionality** for data analysis

**Start exploring emotional AI with your trained probes! 🚀**
