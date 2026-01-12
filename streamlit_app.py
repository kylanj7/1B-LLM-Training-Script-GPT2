"""
Streamlit Chatbot App for Testing Your Trained 1B LLM

This app provides an interactive chat interface to test your trained model.

Usage:
    streamlit run chatbot_app.py
"""

import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import time

# ============================================================================
# Configuration
# ============================================================================

class ChatbotConfig:
    """Configuration for the chatbot"""
    
    # Model path - change this to your trained model
    model_path = "./outputs/best_model.pt"  # or "final_model.pt"
    
    # Model architecture (must match training config)
    vocab_size = 50257
    n_positions = 1024
    n_embd = 1536
    n_layer = 24
    n_head = 16
    n_inner = 6144
    
    # Generation parameters
    max_length = 200          # Max tokens to generate
    temperature = 0.8         # Randomness (0.1=conservative, 1.5=creative)
    top_k = 50               # Consider top K tokens
    top_p = 0.9              # Nucleus sampling
    repetition_penalty = 1.2  # Penalize repetition
    
    # Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Model Loading
# ============================================================================

@st.cache_resource
def load_model(config: ChatbotConfig):
    """
    Load the trained model (cached for performance)
    
    Args:
        config: ChatbotConfig instance
    
    Returns:
        model, tokenizer
    """
    with st.spinner("Loading model... This may take a minute..."):
        # Create model architecture
        model_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_inner=config.n_inner,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
        
        model = GPT2LMHeadModel(model_config)
        
        # Load trained weights
        state_dict = torch.load(config.model_path, map_location=config.device)
        model.load_state_dict(state_dict)
        
        model.to(config.device)
        model.eval()
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        st.success(f"✓ Model loaded successfully on {config.device}")
        
        return model, tokenizer


# ============================================================================
# Text Generation
# ============================================================================

def generate_response(
    model,
    tokenizer,
    prompt: str,
    config: ChatbotConfig
) -> str:
    """
    Generate a response from the model
    
    Args:
        model: Trained GPT2 model
        tokenizer: GPT2 tokenizer
        prompt: Input text
        config: Generation configuration
    
    Returns:
        Generated text
    """
    # Encode input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + config.max_length,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1
        )
    
    # Decode output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Remove the prompt from the output
    response = generated_text[len(prompt):].strip()
    
    return response


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    """Main Streamlit app"""
    
    # Page configuration
    st.set_page_config(
        page_title="1B LLM Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stTextInput > div > div > input {
            font-size: 16px;
        }
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            align-items: flex-end;
        }
        .bot-message {
            background-color: #f5f5f5;
            align-items: flex-start;
        }
        .message-author {
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🤖 1B Parameter LLM Chatbot")
    st.markdown("Chat with your trained language model!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        config = ChatbotConfig()
        
        # Model path
        st.text_input(
            "Model Path",
            value=config.model_path,
            key="model_path",
            help="Path to your trained model weights"
        )
        config.model_path = st.session_state.model_path
        
        st.markdown("---")
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        config.temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1,
            help="Higher = more random, Lower = more focused"
        )
        
        config.max_length = st.slider(
            "Max Length",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="Maximum tokens to generate"
        )
        
        config.top_k = st.slider(
            "Top K",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
            help="Consider top K most likely tokens"
        )
        
        config.top_p = st.slider(
            "Top P (Nucleus)",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Cumulative probability threshold"
        )
        
        config.repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.2,
            step=0.1,
            help="Penalize repeated tokens"
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("ℹ️ Model Info")
        st.info(f"""
        **Parameters:** ~1B  
        **Device:** {config.device}  
        **Architecture:** GPT-2 style  
        **Layers:** {config.n_layer}
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Load model
    try:
        model, tokenizer = load_model(config)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure the model path is correct and the model file exists.")
        st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-author">👤 You</div>
                    <div>{content}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-author">🤖 Assistant</div>
                    <div>{content}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-author">👤 You</div>
                <div>{user_input}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                # Build context from chat history
                context = ""
                for msg in st.session_state.messages[-5:]:  # Last 5 messages
                    if msg["role"] == "user":
                        context += f"User: {msg['content']}\n"
                    else:
                        context += f"Assistant: {msg['content']}\n"
                
                context += "Assistant:"
                
                # Generate
                response = generate_response(model, tokenizer, context, config)
                
                # Add assistant response to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Display assistant response
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-author">🤖 Assistant</div>
                        <div>{response}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
        
        # Rerun to update chat
        st.rerun()
    
    # Example prompts
    if len(st.session_state.messages) == 0:
        st.markdown("### 💡 Try these example prompts:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Write a story"):
                st.session_state.example_prompt = "Write a short story about"
        
        with col2:
            if st.button("🔬 Explain a concept"):
                st.session_state.example_prompt = "Explain quantum computing in simple terms:"
        
        with col3:
            if st.button("💭 Creative writing"):
                st.session_state.example_prompt = "Complete this sentence: The mysterious door opened and"


# ============================================================================
# Alternative: Simple Text Completion Interface
# ============================================================================

def simple_completion_app():
    """Alternative simpler interface for text completion"""
    
    st.set_page_config(
        page_title="LLM Text Completion",
        page_icon="✍️",
        layout="wide"
    )
    
    st.title("✍️ Text Completion with Your 1B LLM")
    
    config = ChatbotConfig()
    
    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        config.temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
        config.max_length = st.slider("Max Tokens", 50, 500, 200, 50)
    
    # Load model
    model, tokenizer = load_model(config)
    
    # Input
    prompt = st.text_area(
        "Enter your prompt:",
        height=150,
        placeholder="Once upon a time..."
    )
    
    # Generate button
    if st.button("Generate", type="primary"):
        if prompt:
            with st.spinner("Generating..."):
                response = generate_response(model, tokenizer, prompt, config)
                
                st.markdown("### Generated Text:")
                st.success(prompt + response)
                
                # Copy button
                st.code(prompt + response, language=None)
        else:
            st.warning("Please enter a prompt first!")


# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
    # Choose which interface to use
    # Option 1: Chat interface (default)
    main()
    
    # Option 2: Simple completion interface (uncomment to use)
    # simple_completion_app()
