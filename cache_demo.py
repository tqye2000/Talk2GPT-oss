"""
Demo script to show model caching benefits in Streamlit
"""

import streamlit as st
import time
from datetime import datetime

st.set_page_config(
    page_title="🚀 Model Caching Demo",
    page_icon="🚀",
)

st.title("🚀 Model Caching Benefits Demo")

st.markdown("""
### 🎯 **Model Caching with `@st.cache_resource`**

The GPT-OSS-20B chatbot now uses Streamlit's `@st.cache_resource` decorator for optimal performance:

#### ✅ **Benefits:**
- **One-time Loading**: Model loads only once per Streamlit session
- **Faster Reloads**: Page refreshes use cached model instantly
- **Memory Efficient**: Avoids multiple model instances
- **Better UX**: No waiting for model reload on every interaction

#### 🔄 **How it Works:**
1. **First Run**: Model loads and gets cached (takes 2-5 minutes)
2. **Subsequent Runs**: Uses cached model (instant!)
3. **Browser Refresh**: Still uses cached model
4. **Server Restart**: Cache clears, model reloads once

#### 💡 **Cache Management:**
- View cache status in the sidebar
- Clear cache manually if needed
- Automatic cache invalidation on model path changes
""")

st.divider()

# Simple cache demo
@st.cache_resource
def expensive_operation(operation_name):
    """Simulate expensive model loading"""
    with st.spinner(f"Performing {operation_name}... (simulated delay)"):
        time.sleep(2)  # Simulate loading time
    return f"✅ {operation_name} completed at {datetime.now().strftime('%H:%M:%S')}"

st.subheader("🧪 **Cache Demo**")

col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Run Expensive Operation"):
        result = expensive_operation("Model Loading Simulation")
        st.success(result)

with col2:
    # Show cache status (note: cache_resource doesn't have get_stats)
    st.info("Cache status: Check terminal output for cache behavior")
    
    if st.button("🧹 Clear Demo Cache"):
        expensive_operation.clear()
        st.info("Cache cleared! Next operation will be slow again.")

st.info("👆 **Try this**: Click 'Run Expensive Operation' multiple times. First click is slow (2s delay), subsequent clicks are instant!")

st.divider()

st.markdown("""
### 🎮 **Try the Real Chatbot:**
Run the main chatbot with: `streamlit run streamlit_chatbot.py`

**First time**: Wait 2-5 minutes for model loading  
**After that**: Instant startup! 🚀
""")
