# Talk2GPT-oss: Advanced Streamlit Interface for GPT-OSS Models

A sophisticated Streamlit application for interacting with locally hosted GPT-OSS models, featuring intelligent response processing, memory management, and educational tools.

## ✨ Key Features

### 🧠 **Intelligent Response Display**
- **Clean Answers**: Automatically filters out internal reasoning for professional responses
- **Optional Reasoning View**: Expandable sections showing the model's thought process
- **Educational Mode**: Toggle to show/hide internal reasoning for learning purposes
- **Smart Detection**: Automatically identifies and processes various reasoning patterns

### 🔧 **Advanced Memory Management**
- **CUDA Memory Optimization**: Intelligent GPU memory cleanup and monitoring
- **Page Reload Protection**: Automatic memory cleanup on session changes
- **Emergency Tools**: Multiple cleanup options for memory issues
- **Real-time Monitoring**: GPU memory usage display with warnings

### 💬 **Enhanced Chat Experience**
- **Session Management**: Persistent conversations with cleanup options
- **Export Functionality**: Download conversation history
- **Statistics Dashboard**: Message counts and memory usage
- **Responsive Design**: Clean, professional interface

## Download the model
```bash
git clone https://huggingface.co/openai/gpt-oss-120b
git clone https://huggingface.co/openai/gpt-oss-20b
```

## Install Requirements
```bash
pip install -r requirements.txt
```

## Quick Start
```bash
# For console mode:
python test_gpt-oss-20B.py 

# For the advanced Streamlit interface:
streamlit run app.py --server.port 8501
```

## Using the Reasoning Display Feature

### Example Interaction
**Your Question**: "Hand is to arm as foot is to ___"

**Clean Response**: "leg"

**Reasoning (in expander)**: 
```
Analysis: We have a classic analogy: Hand is to arm as foot is to??? 
We need to find the correct word. Hand is part of arm. Foot is part 
of leg. So answer: leg...
```

### Controls
- **Sidebar Toggle**: "🧠 Show Internal Reasoning" checkbox
- **Enabled**: Shows reasoning in expandable sections
- **Disabled**: Shows only clean responses with optional indicator

## Memory Management

### CUDA Out of Memory Issues
If you encounter CUDA out of memory errors, especially when reloading the page, use the following solutions:

#### Quick Solutions (In the Streamlit App):
1. **Emergency Cleanup Button**: Use the "🚨 Emergency Cleanup" button in the sidebar
2. **Clear GPU Cache**: Use the "🧹 Clear GPU Cache" button
3. **Clear Conversation History**: Use the "🧹 Clear History" button to free up context memory
4. **Reduce Max Tokens**: Lower the "Max Tokens per Response" slider

#### Best Practices:
1. Clear conversation history regularly for long sessions
2. Use the Emergency Cleanup if you see memory warnings
3. Monitor GPU memory usage in the sidebar
4. Restart the application if memory issues persist
5. Consider reducing max tokens for memory-constrained systems

#### Troubleshooting:
- If the app crashes on startup, run `memory_cleanup.ps1` first
- For persistent issues, restart your computer to fully clear GPU memory
- Check that you have at least 2-3 GB of free GPU memory before starting 