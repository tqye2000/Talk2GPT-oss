##################################################################
# Streamlit application for interacting with GPT-OSS-20B model
#
# History:
# 2025-08-13| TQ Ye         |- Initial version created| 
###################################################################
import streamlit as st
from transformers import pipeline
import torch
import gc
import os
import sys
from datetime import datetime
import time
import atexit

MODEL_PATH = r"D:\hf_models\gpt-oss-20b"

def cleanup_gpu_memory():
    """Comprehensive GPU memory cleanup."""
    if torch.cuda.is_available():
        # Clear all cached tensors
        torch.cuda.empty_cache()
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Additional memory cleanup
        if hasattr(torch.cuda, 'reset_max_memory_allocated'):
            torch.cuda.reset_max_memory_allocated()
        
        # Clear all CUDA streams
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def clear_model_from_memory():
    """Clear model and associated resources from memory."""
    # Clear cached resource if it exists
    if hasattr(load_gpt_model, 'clear'):
        load_gpt_model.clear()
    
    # Clear session state chatbot
    if "chatbot" in st.session_state:
        chatbot = st.session_state.chatbot
        if hasattr(chatbot, 'pipe') and chatbot.pipe is not None:
            # Clear the pipeline model
            if hasattr(chatbot.pipe, 'model'):
                del chatbot.pipe.model
            if hasattr(chatbot.pipe, 'tokenizer'):
                del chatbot.pipe.tokenizer
            del chatbot.pipe
        del st.session_state.chatbot
    
    # Clear model cache timestamp
    if "model_cached_at" in st.session_state:
        del st.session_state.model_cached_at
    
    # Comprehensive memory cleanup
    cleanup_gpu_memory()

# Register cleanup function to be called on exit
atexit.register(cleanup_gpu_memory)

@st.cache_resource
def load_gpt_model(model_path, max_new_tokens=512):
    """Load and cache the GPT model to avoid repeated loading."""
    # Comprehensive memory cleanup before loading
    cleanup_gpu_memory()
    
    # Set environment variables for better memory management
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For better error reporting
    
    try:
        with st.spinner("Loading GPT-OSS-20B model... This may take a few minutes."):
            # Load with specific memory-efficient settings
            pipe = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto",
                max_memory={0: "40GiB"},  # Limit memory usage to leave some free
                low_cpu_mem_usage=True,
                pad_token_id=50256  # Set pad token to avoid warnings
            )
            
            # Move to eval mode to save memory
            if hasattr(pipe.model, 'eval'):
                pipe.model.eval()
        
        st.success("✅ Model loaded successfully and cached!")
        
        # Store cache timestamp in session state for tracking
        if "model_cached_at" not in st.session_state:
            st.session_state.model_cached_at = datetime.now()
        
        return pipe
    except Exception as e:
        # Clean up on error
        cleanup_gpu_memory()
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("💡 Try clearing GPU cache or restarting the application if you're experiencing memory issues.")
        st.stop()

class StreamlitGPTChatbot:
    def __init__(self, model_path, max_new_tokens=512):
        """Initialize the GPT chatbot with the specified model."""
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        
        # Use the cached model loading function
        self.pipe = load_gpt_model(model_path, max_new_tokens)
    
    # Remove the _load_model method as it's now handled by the cached function
    
    def get_response(self, conversation_history, user_input):
        """Generate a response from the model."""
        # Prepare the full conversation including the new user input
        messages = conversation_history + [{"role": "user", "content": user_input}]
        
        # Keep last 15 messages to manage memory (reduced from 20)
        recent_messages = messages[-15:]
        
        try:
            # Clear GPU cache before generation to ensure maximum available memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            with st.spinner("Generating response..."):
                outputs = self.pipe(
                    recent_messages,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    # Additional memory-efficient settings
                    clean_up_tokenization_spaces=True,
                    return_full_text=False  # Only return new tokens
                )
            
            # Extract the assistant's response
            generated_text = outputs[0]["generated_text"]
            if isinstance(generated_text, list):
                # Find the last assistant message
                assistant_response = None
                for msg in reversed(generated_text):
                    if msg.get("role") == "assistant":
                        assistant_response = msg["content"]
                        break
                
                if assistant_response is None:
                    # If no assistant message found, take the last message content
                    assistant_response = generated_text[-1]["content"] if generated_text else "I'm sorry, I couldn't generate a response."
            else:
                assistant_response = generated_text
            
            # Clean up after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            return assistant_response
            
        except torch.cuda.OutOfMemoryError as oom_error:
            # Specific handling for CUDA OOM errors
            st.error("🚨 CUDA Out of Memory Error!")
            st.error("💡 Try clearing GPU cache, reducing max tokens, or clearing conversation history.")
            
            # Aggressive cleanup
            cleanup_gpu_memory()
            
            return f"I apologize, but I ran out of GPU memory. Please try clearing the GPU cache or reducing the conversation length."
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            
            # Clean up on any error
            if torch.cuda.is_available():
                cleanup_gpu_memory()
            
            return f"Sorry, I encountered an error: {str(e)}"

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chatbot" not in st.session_state:
        
        # Check if model path exists
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Error: Model path not found: {MODEL_PATH}")
            st.error("Please ensure the model is downloaded and the path is correct.")
            st.stop()
        
        # Initialize chatbot with cached model loading
        # The model will only be loaded once and cached for subsequent runs
        st.session_state.chatbot = StreamlitGPTChatbot(MODEL_PATH)
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 256

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="🤖 Ask GPT-OSS-20B",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Model information
        st.info(f"**Model:** GPT-OSS-20B\n**Path:** {st.session_state.chatbot.model_path}")
        
        # Cache management
        st.subheader("🗄️ Cache Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear Model Cache", use_container_width=True, help="Clear cached model to force reload"):
                load_gpt_model.clear()
                # Clear the cached timestamp as well
                if "model_cached_at" in st.session_state:
                    del st.session_state.model_cached_at
                st.success("Model cache cleared!")
                st.info("Please refresh the page to reload the model.")
        
        with col2:
            # Check if the model is cached and show cache time
            try:
                if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot.pipe is not None:
                    if "model_cached_at" in st.session_state:
                        cached_time = st.session_state.model_cached_at.strftime("%H:%M:%S")
                        st.success(f"✅ Model Cached\nSince: {cached_time}")
                    else:
                        st.success("✅ Model Cached")
                else:
                    st.info("ℹ️ No Cache")
            except:
                st.info("ℹ️ No Cache")
        
        st.divider()
        
        # Max tokens slider
        st.subheader("🎯 Generation Settings")
        new_max_tokens = st.slider(
            "Max Tokens per Response",
            min_value=50,
            max_value=1024,
            value=st.session_state.max_tokens,
            step=50,
            help="Maximum number of tokens the model will generate"
        )
        
        if new_max_tokens != st.session_state.max_tokens:
            st.session_state.max_tokens = new_max_tokens
            st.session_state.chatbot.max_new_tokens = new_max_tokens
            st.success(f"Max tokens updated to {new_max_tokens}")
        
        st.divider()
        
        # Conversation management
        st.subheader("📝 Conversation")
        
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state.messages = []
            gc.collect()
            st.success("Conversation history cleared!")
            st.rerun()
        
        if st.button("💾 Export History", use_container_width=True):
            if st.session_state.messages:
                # Create a text export of the conversation
                export_text = f"GPT-OSS-20B Conversation Export\n"
                export_text += f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                export_text += "=" * 50 + "\n\n"
                
                for i, message in enumerate(st.session_state.messages, 1):
                    role_icon = "👤 User" if message["role"] == "user" else "🤖 Assistant"
                    export_text += f"{i}. {role_icon}:\n{message['content']}\n\n"
                
                st.download_button(
                    label="📥 Download Conversation",
                    data=export_text,
                    file_name=f"gpt_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No conversation to export!")
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("Assistant Messages", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
        
        # GPU Memory info (if available)
        if torch.cuda.is_available():
            st.divider()
            st.subheader("🖥️ GPU Memory")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_used = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_free = gpu_memory - gpu_used
            
            st.metric("Total GPU Memory", f"{gpu_memory:.1f} GB")
            st.metric("Used GPU Memory", f"{gpu_used:.1f} GB")
            st.metric("Free GPU Memory", f"{gpu_free:.1f} GB")
            
            if st.button("🧹 Clear GPU Cache", use_container_width=True):
                torch.cuda.empty_cache()
                gc.collect()
                st.success("GPU cache cleared!")
                st.rerun()
    
    # Main chat interface
    st.title("🤖 Ask GPT-OSS-20B")
    st.markdown("Welcome to the GPT-OSS chatbot! Ask me anything and I'll do my best to help.")
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Get response from the model
            response = st.session_state.chatbot.get_response(
                st.session_state.messages[:-1],  # Exclude the last user message as it's added in the method
                prompt
            )
            
            # Display the response
            response_placeholder.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
