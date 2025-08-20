#########################################################################
# Streamlit application for interacting with GPT-OSS-20B model
#
# History:
# 2025-08-13| TQ Ye         |- Initial version created| 
# 2025-08-16| TQ Ye         |- Added comprehensive GPU memory cleanup
#                           |- Better handle the responses from the model
#########################################################################
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
            # Load with memory-efficient settings
            pipe = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype="auto",  # Let the model use its preferred dtype
                device_map="auto"
            )
            
            # Set pad token if not already set
            if pipe.tokenizer.pad_token is None:
                pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
            
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
                    pad_token_id=self.pipe.tokenizer.eos_token_id
                )
            
            # Extract the assistant's response
            generated_text = outputs[0]["generated_text"]
            
            # Process the response based on its format
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
                # Handle raw text generation - this is where the issue occurs
                assistant_response = generated_text
            
            # Clean up the response to remove internal reasoning patterns
            cleaned_response, reasoning_content = self._clean_response(assistant_response)
            
            # Clean up after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Return both the cleaned response and reasoning content for display
            return cleaned_response, reasoning_content
            
        except torch.cuda.OutOfMemoryError as oom_error:
            # Specific handling for CUDA OOM errors
            st.error("🚨 CUDA Out of Memory Error!")
            st.error("💡 Try clearing GPU cache, reducing max tokens, or clearing conversation history.")
            
            # Aggressive cleanup
            cleanup_gpu_memory()
            
            error_msg = f"I apologize, but I ran out of GPU memory. Please try clearing the GPU cache or reducing the conversation length."
            return error_msg, None
            
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            
            # Clean up on any error
            if torch.cuda.is_available():
                cleanup_gpu_memory()
            
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            return error_msg, None
    
    def _clean_response(self, response):
        """Clean up the model response to remove internal reasoning patterns."""
        if not isinstance(response, str):
            return str(response), None
        
        # Remove common internal reasoning patterns
        import re
        
        original_response = response
        cleaned_response = response
        reasoning_content = None
        
        # Check if this response contains internal reasoning
        reasoning_detected = any(marker in original_response.lower() for marker in 
                               ['analysis', 'final:', 'answer:', 'assistantfinal', 'let me think', 'i need to analyze', 
                                'the user\'s question:', 'provide explanation:', 'provide steps:', 'provide details:'])
        
        if reasoning_detected:
            # For new structured format, extract everything before the actual answer
            if 'the user\'s question:' in original_response.lower():
                # Find the start of the actual answer (after all the "provide" sections)
                lines = original_response.split('\n')
                answer_start_idx = -1
                
                for i, line in enumerate(lines):
                    line_lower = line.lower().strip()
                    # Look for lines that start the actual calculation/answer
                    if (re.match(r'^\d+\s*[÷/×+\-]\s*\d+', line.strip()) or
                        'therefore' in line_lower or
                        'so ' in line_lower and ('=' in line or 'times' in line_lower) or
                        line.strip().startswith('The answer is') or
                        line.strip().startswith('Answer:')):
                        answer_start_idx = i
                        break
                
                if answer_start_idx > 0:
                    # Everything before the answer is reasoning
                    reasoning_content = '\n'.join(lines[:answer_start_idx]).strip()
                    # Everything from the answer onwards is the cleaned response
                    cleaned_response = '\n'.join(lines[answer_start_idx:]).strip()
                else:
                    # Fallback: extract the user's question line as reasoning
                    reasoning_match = re.search(r'(the user\'s question:.*?)(?=\n\n)', original_response, re.IGNORECASE | re.DOTALL)
                    if reasoning_match:
                        reasoning_content = reasoning_match.group(1).strip()
            
            # Apply other cleaning patterns for older formats
            else:
                # Extract reasoning content before cleaning
                reasoning_patterns = [
                    r'(analysis.*?assistantfinal)',
                    r'(analysis:.*?(?:final:|answer:|response:))',
                    r'(let me think.*?)(?=\n\n|\.|!|\?|$)',
                    r'(i need to analyze.*?)(?=\n\n|\.|!|\?|$)',
                ]
                
                for pattern in reasoning_patterns:
                    match = re.search(pattern, original_response, re.IGNORECASE | re.DOTALL)
                    if match:
                        reasoning_content = match.group(1).strip()
                        break
                
                # Pattern 1: Remove "analysis...assistant...final..." patterns
                analysis_pattern = r'analysis.*?assistantfinal\s*'
                cleaned_response = re.sub(analysis_pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Pattern 2: Remove "analysis:" followed by reasoning until "final:" or "answer:"
                analysis_colon_pattern = r'analysis:.*?(?:final:|answer:|response:)\s*'
                cleaned_response = re.sub(analysis_colon_pattern, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Pattern 3: Remove anything that looks like internal reasoning tags
                internal_tags = r'<[^>]*thinking[^>]*>.*?</[^>]*thinking[^>]*>'
                cleaned_response = re.sub(internal_tags, '', cleaned_response, flags=re.DOTALL | re.IGNORECASE)
                
                # Pattern 4: Remove "Let me think..." or "I need to analyze..." beginnings
                thinking_pattern = r'^(Let me think|I need to analyze|Let me analyze|I should consider).*?(?:\.|!|\?)\s*'
                cleaned_response = re.sub(thinking_pattern, '', cleaned_response, flags=re.IGNORECASE)
                
                # Pattern 8: Extract answer after "final" or "answer:" keywords
                final_answer_patterns = [
                    r'(?:final|answer|response):\s*(.+?)(?:\n|$)',
                    r'(?:final|answer|response)\s+(.+?)(?:\n|$)',
                    r'assistantfinal\s*(.+?)(?:\n|$)'
                ]
                
                for pattern in final_answer_patterns:
                    match = re.search(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    if match:
                        extracted = match.group(1).strip()
                        if extracted and len(extracted) > 0:
                            cleaned_response = extracted
                            break
        
        # Clean up whitespace and formatting
        cleaned_response = cleaned_response.strip()
        
        # Remove extra newlines and spaces
        cleaned_response = re.sub(r'\s+', ' ', cleaned_response)
        
        # If the response is empty or too short after cleaning, provide a fallback
        if not cleaned_response or len(cleaned_response.strip()) < 2:
            cleaned_response = "I apologize, but I couldn't generate a clear response. Could you please rephrase your question?"
            reasoning_content = None
        
        return cleaned_response, reasoning_content

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    # Check if this is a fresh session (page reload detection)
    if "session_id" not in st.session_state:
        # Generate a new session ID
        st.session_state.session_id = f"session_{int(time.time())}"
        
        # Clear any existing GPU memory from previous sessions
        cleanup_gpu_memory()
        
        # Clear any cached resources if this is a reload
        if hasattr(load_gpt_model, 'clear'):
            load_gpt_model.clear()
    
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
        try:
            st.session_state.chatbot = StreamlitGPTChatbot(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Failed to initialize chatbot: {str(e)}")
            st.error("💡 Try refreshing the page or clearing browser cache if the issue persists.")
            # Attempt cleanup and stop
            cleanup_gpu_memory()
            st.stop()
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 850
    
    if "show_reasoning" not in st.session_state:
        st.session_state.show_reasoning = True

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
    
    # Add session info for debugging
    if st.sidebar.checkbox("Show Debug Info", value=False):
        with st.sidebar.expander("🔧 Debug Information"):
            st.write(f"Session ID: {st.session_state.get('session_id', 'Unknown')}")
            st.write(f"Messages count: {len(st.session_state.messages)}")
            if "model_cached_at" in st.session_state:
                st.write(f"Model cached at: {st.session_state.model_cached_at}")
            
            # Show actual GPU memory if available
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    st.write(f"GPU Allocated: {allocated:.2f} GB")
                    st.write(f"GPU Reserved: {reserved:.2f} GB")
                except:
                    st.write("Could not read GPU memory")
    
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
                clear_model_from_memory()
                st.success("Model cache and GPU memory cleared!")
                st.info("Please refresh the page to reload the model.")
        
        with col2:
            if st.button("🧹 Clear GPU Cache", use_container_width=True, help="Clear GPU cache without removing model"):
                cleanup_gpu_memory()
                st.success("GPU cache cleared!")
                st.rerun()
        
        # Emergency memory cleanup
        if st.button("🚨 Emergency Cleanup", use_container_width=True, 
                    help="Comprehensive memory cleanup for out of memory issues", 
                    type="secondary"):
            try:
                # Clear everything
                clear_model_from_memory()
                
                # Clear conversation history
                if "messages" in st.session_state:
                    st.session_state.messages = []
                
                # Force cleanup
                cleanup_gpu_memory()
                
                st.success("Emergency cleanup completed!")
                st.info("All caches cleared. Please refresh the page.")
                
            except Exception as e:
                st.error(f"Error during emergency cleanup: {e}")
        
        # Cache status display
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
        
        # Show reasoning toggle
        show_reasoning = st.checkbox(
            "🧠 Show Internal Reasoning",
            value=st.session_state.show_reasoning,
            help="Display the model's internal thought process in an expander"
        )
        
        if show_reasoning != st.session_state.show_reasoning:
            st.session_state.show_reasoning = show_reasoning
            if show_reasoning:
                st.success("✅ Will show reasoning process in expanders")
            else:
                st.info("ℹ️ Reasoning process will be hidden")
        
        st.divider()
        
        # Conversation management
        st.subheader("📝 Conversation")
        
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state.messages = []
            # Clear GPU cache after clearing history to free up memory
            cleanup_gpu_memory()
            st.success("Conversation history cleared and GPU cache cleaned!")
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
            
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                gpu_free = gpu_memory - gpu_reserved
                
                # Color coding for memory usage
                usage_percentage = (gpu_reserved / gpu_memory) * 100
                
                st.metric("Total GPU Memory", f"{gpu_memory:.1f} GB")
                st.metric("Allocated Memory", f"{gpu_allocated:.1f} GB")
                st.metric("Reserved Memory", f"{gpu_reserved:.1f} GB")
                st.metric("Free Memory", f"{gpu_free:.1f} GB")
                
                # Memory usage bar
                st.progress(usage_percentage / 100, text=f"Memory Usage: {usage_percentage:.1f}%")
                
                # Warning for high memory usage
                if usage_percentage > 85:
                    st.warning("⚠️ High memory usage! Consider clearing cache or history.")
                elif usage_percentage > 95:
                    st.error("🚨 Critical memory usage! Clear cache immediately!")
                
            except Exception as e:
                st.error(f"Error reading GPU memory: {e}")
                
        else:
            st.divider()
            st.info("🖥️ No CUDA GPU detected")
    
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
            
            # Get response from the model (now returns cleaned response and reasoning)
            response_data = st.session_state.chatbot.get_response(
                st.session_state.messages[:-1],  # Exclude the last user message as it's added in the method
                prompt
            )
            
            # Handle the new tuple format
            if isinstance(response_data, tuple):
                cleaned_response, reasoning_content = response_data
            else:
                # Fallback for backward compatibility
                cleaned_response = response_data
                reasoning_content = None
            
            # Display the cleaned response
            response_placeholder.markdown(cleaned_response)
            
            # Display reasoning in an expander if available and enabled
            if (reasoning_content and reasoning_content.strip() and 
                st.session_state.get("show_reasoning", True)):
                with st.expander("🧠 View Internal Reasoning Process", expanded=False):
                    st.markdown("**Model's thought process:**")
                    # Format the reasoning content for better readability
                    formatted_reasoning = reasoning_content.replace("analysis", "**Analysis:**").replace("assistantfinal", "\n\n**Final Answer:**")
                    st.markdown(formatted_reasoning)
                    st.caption("This shows how the model arrived at its answer.")
            elif reasoning_content and reasoning_content.strip():
                # Show a small indicator that reasoning is available but hidden
                st.caption("🧠 Internal reasoning available (enable in sidebar to view)")
            
            # Add assistant response to chat history (only the cleaned response)
            st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

if __name__ == "__main__":
    main()
