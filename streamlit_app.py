"""FinanceGPT - AI-Powered Financial Advisory Chatbot"""

import streamlit as st
import tensorflow as tf
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import pandas as pd
import time
from datetime import datetime
import json
import os
from pathlib import Path

st.set_page_config(
    page_title="FinanceGPT - AI Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main {
        padding: 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0 0.5rem auto;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
        animation: slideInRight 0.3s ease-out;
        margin-left: auto;
        display: block;
        width: fit-content;
    }
    
    .bot-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #333;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem auto 0.5rem 0;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        animation: slideInLeft 0.3s ease-out;
        display: block;
        width: fit-content;
    }
    
    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        margin-right: 10px;
        vertical-align: middle;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 2px solid rgba(255, 255, 255, 0.8);
    }
    
    .bot-avatar {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border: 2px solid rgba(255, 255, 255, 0.8);
    }
    
    .message-row {
        display: flex;
        align-items: flex-start;
        margin: 1rem 0;
        gap: 10px;
    }
    
    .message-row.user {
        flex-direction: row-reverse;
    }
    
    .message-content {
        flex: 1;
    }
    
    .typing-indicator {
        background: #f5f7fa;
        padding: 1rem 1.5rem;
        border-radius: 18px;
        display: inline-block;
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes slideInRight {
        from {
            transform: translateX(20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideInLeft {
        from {
            transform: translateX(-20px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .header-container h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .header-container p {
        font-size: 0.95rem;
        opacity: 0.95;
        margin: 0;
    }
    
    .faq-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .faq-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2);
    }
    
    .faq-question {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .faq-question:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateX(10px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .big-emoji {
        font-size: 3rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stTextInput input {
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
    }
    
    .stTextInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .stForm {
        border: none !important;
        padding: 0 !important;
    }
    
    div[data-testid="column"]:has(button[kind="formSubmit"]) button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        height: 100% !important;
    }
    
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    .chat-container::-webkit-scrollbar-thumb:hover {
        background: #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_path):
    """Load T5 model and tokenizer with caching."""
    try:
        # Configure GPU if available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load tokenizer (will download from HF Hub if needed)
        st.info(f"Loading tokenizer from: {model_path}")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        # Try multiple loading strategies
        st.info("Loading model... This may take a minute on first run.")
        try:
            # Try TensorFlow format first
            model = TFT5ForConditionalGeneration.from_pretrained(
                model_path,
                from_pt=False
            )
        except Exception as e1:
            st.warning("TensorFlow format not found, trying PyTorch conversion...")
            try:
                # Try converting from PyTorch
                model = TFT5ForConditionalGeneration.from_pretrained(
                    model_path,
                    from_pt=True
                )
            except Exception as e2:
                st.error(f"Failed to load model: {str(e2)}")
                st.info("""
                **Troubleshooting:**
                - Model: `{}`
                - Make sure the model repository exists and is public
                - Check your internet connection for first-time download
                """.format(model_path))
                return None, None
        
        st.success(f"✅ Model loaded successfully from {model_path}")
        return model, tokenizer
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("""
        **Troubleshooting Tips:**
        - Make sure the model path is correct: `{}`
        - Verify the model exists at: https://huggingface.co/{}
        - Check that you have internet connection (first download)
        - Ensure TensorFlow and transformers are installed
        """.format(model_path, model_path))
        return None, None

def generate_answer(
    question, 
    model, 
    tokenizer, 
    max_length=512,
    num_beams=1,
    temperature=0.7,
    no_repeat_ngram_size=2
):
    """Generate answer for a given question using the T5 model."""
    try:
        start_time = time.time()
        
        input_text = f"question: {question}"
        input_ids = tokenizer(
            input_text, 
            return_tensors="tf", 
            max_length=128,
            truncation=True,
            padding=False
        ).input_ids
        
        generation_config = {
            "max_length": max_length,
            "min_length": 30,  # Reduced from 50 for flexibility
            "num_beams": num_beams,
            "early_stopping": True,
            "no_repeat_ngram_size": 2,
            "length_penalty": 0.8,  # Slightly reduced to allow longer outputs
            "repetition_penalty": 1.3,  # Reduced from 1.5 to allow necessary repetition
        }
        
        # Use sampling for faster generation (num_beams=1)
        if num_beams == 1:
            generation_config.update({
                "do_sample": True,
                "temperature": temperature,
                "top_k": 50,
                "top_p": 0.95,  # Increased for more diverse outputs
            })
        
        outputs = model.generate(input_ids, **generation_config)
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        confidence = estimate_confidence(answer, question)
        
        if confidence < 0.3:
            answer += "\n\n⚠️ **Disclaimer**: I don't have complete information on this specific topic. Please consult professional financial resources or advisors for accurate guidance."
        elif confidence < 0.5:
            answer += "\n\n💡 **Note**: My knowledge on this topic may be limited. For comprehensive information, consider consulting additional financial sources."
        
        return answer, generation_time, confidence
    
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return f"Sorry, I encountered an error: {str(e)}", 0, 0.0

def estimate_confidence(answer, question):
    """Estimate confidence based on answer quality indicators."""
    score = 0.5
    
    words = answer.split()
    word_count = len(words)
    
    if word_count < 15:
        score -= 0.3
    elif word_count < 30:
        score -= 0.1
    elif word_count > 60:
        score += 0.15
    
    if word_count > 0:
        unique_words = len(set(words))
        uniqueness_ratio = unique_words / word_count
        
        if uniqueness_ratio < 0.4:
            score -= 0.4
        elif uniqueness_ratio < 0.6:
            score -= 0.2
        elif uniqueness_ratio > 0.8:
            score += 0.1
    
    uncertainty_phrases = [
        "i don't know", "i'm not sure", "i don't have", "uncertain",
        "unclear", "not entirely clear", "difficult to say"
    ]
    if any(phrase in answer.lower() for phrase in uncertainty_phrases):
        score -= 0.3
    
    financial_terms = [
        "interest rate", "credit score", "credit report", "debt", "investment",
        "stock", "bond", "portfolio", "asset", "liability", "equity",
        "dividend", "capital", "mortgage", "loan", "apr", "apy", "savings",
        "budget", "tax", "fico", "compound interest", "principal", "collateral"
    ]
    term_matches = sum(1 for term in financial_terms if term in answer.lower())
    score += min(0.25, term_matches * 0.05)
    
    question_keywords = set(word.lower().strip('?.,!') for word in question.split() if len(word) > 3)
    answer_words_lower = set(word.lower().strip('?.,!') for word in words if len(word) > 3)
    
    if question_keywords:
        overlap = len(question_keywords & answer_words_lower)
        overlap_ratio = overlap / len(question_keywords)
        
        if overlap_ratio > 0.5:
            score += 0.15
        elif overlap_ratio < 0.2:
            score -= 0.15
    
    generic_phrases = [
        "as mentioned", "in conclusion", "in summary", "to sum up",
        "it is important", "you should know", "keep in mind"
    ]
    generic_count = sum(1 for phrase in generic_phrases if phrase in answer.lower())
    if generic_count > 2:
        score -= 0.1
    
    sentence_markers = answer.count('.') + answer.count('!') + answer.count('?')
    if sentence_markers >= 3:
        score += 0.1
    
    return max(0.0, min(1.0, score))

def initialize_session_state():
    """Initialize session state variables."""
    if 'all_chats' not in st.session_state:
        st.session_state.all_chats = []
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 0
    
    if 'total_response_time' not in st.session_state:
        st.session_state.total_response_time = 0
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    
    if 'tokenizer' not in st.session_state:
        st.session_state.tokenizer = None
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    if 'sample_questions_used' not in st.session_state:
        st.session_state.sample_questions_used = False

def add_to_history(question, answer, response_time, confidence=0.5):
    """Add a Q&A pair to chat history."""
    st.session_state.chat_history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'question': question,
        'answer': answer,
        'response_time': response_time,
        'confidence': confidence
    })
    st.session_state.total_questions += 1
    st.session_state.total_response_time += response_time
    
    save_current_chat()

def is_duplicate_pending(question: str) -> bool:
    """Return True if the last message matches the question and is still pending (typing...)."""
    if not st.session_state.chat_history:
        return False
    last = st.session_state.chat_history[-1]
    return (last.get('question') == question and last.get('answer') == 'typing...')

def save_current_chat():
    """Save current chat to all_chats list."""
    if st.session_state.chat_history and st.session_state.current_chat_id:
        for chat in st.session_state.all_chats:
            if chat['id'] == st.session_state.current_chat_id:
                chat['messages'] = st.session_state.chat_history.copy()
                chat['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if st.session_state.chat_history:
                    first_question = st.session_state.chat_history[0]['question']
                    chat['title'] = first_question[:50] + "..." if len(first_question) > 50 else first_question
                break

def create_new_chat():
    """Create a new chat session."""
    save_current_chat()
    
    new_chat_id = f"chat_{len(st.session_state.all_chats)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chat_history = []
    st.session_state.total_questions = 0
    st.session_state.total_response_time = 0
    
    st.session_state.all_chats.append({
        'id': new_chat_id,
        'title': 'New Chat',
        'messages': [],
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def load_chat(chat_id):
    """Load a specific chat session."""
    save_current_chat()
    
    for chat in st.session_state.all_chats:
        if chat['id'] == chat_id:
            st.session_state.current_chat_id = chat_id
            st.session_state.chat_history = chat['messages'].copy()
            st.session_state.total_questions = len(chat['messages'])
            st.session_state.total_response_time = sum(msg['response_time'] for msg in chat['messages'] if msg.get('response_time', 0) > 0)
            break

def clear_history():
    """Clear current chat history."""
    st.session_state.chat_history = []
    st.session_state.total_questions = 0
    st.session_state.total_response_time = 0
    save_current_chat()

def delete_chat(chat_id):
    """Delete a specific chat session."""
    # Remove chat from all_chats
    st.session_state.all_chats = [
        chat for chat in st.session_state.all_chats if chat['id'] != chat_id
    ]
    
    # If deleted chat was active, switch to another or create new
    if st.session_state.current_chat_id == chat_id:
        if st.session_state.all_chats:
            # Load the most recent chat
            latest_chat = st.session_state.all_chats[-1]
            st.session_state.current_chat_id = latest_chat['id']
            st.session_state.chat_history = latest_chat['messages'].copy()
            st.session_state.total_questions = len(latest_chat['messages'])
            st.session_state.total_response_time = sum(
                msg['response_time'] for msg in latest_chat['messages'] if msg.get('response_time', 0) > 0
            )
        else:
            # No chats left, create a new one
            create_new_chat()

def export_conversation():
    """Export conversation history to JSON."""
    if st.session_state.chat_history:
        return json.dumps(st.session_state.chat_history, indent=2)
    return None

def validate_question(question):
    """Validate user input question."""
    if not question or question.strip() == "":
        return False, "Please enter a question."
    
    if len(question.strip()) < 5:
        return False, "Question is too short. Please provide more details."
    
    if len(question) > 500:
        return False, "Question is too long. Please keep it under 500 characters."
    
    return True, ""

def main():
    """Main application function."""
    
    initialize_session_state()
    
    if not st.session_state.all_chats:
        create_new_chat()
    
    if st.session_state.current_chat_id is None and st.session_state.all_chats:
        st.session_state.current_chat_id = st.session_state.all_chats[0]['id']
        st.session_state.chat_history = st.session_state.all_chats[0]['messages'].copy()
    
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI model... Please wait..."):
            model_path = "GaiusIrakiza/financegpt-t5-model"
            model, tokenizer = load_model_and_tokenizer(model_path)
            if model and tokenizer:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
            else:
                st.error(f"❌ Failed to load model from: {model_path}")
                st.info("Please check the Hugging Face Space logs for more details.")
                st.stop()
    
    # Optimized generation parameters for speed and completeness
    max_length = 1024  # Increased from 768 to prevent truncation
    num_beams = 1  # Changed from 2 to 1 for much faster generation (uses sampling instead)
    temperature = 0.7
    
    st.markdown("""
    <div class="header-container">
        <h1>💰 FinanceGPT - AI Financial Advisor</h1>
        <p style="font-size: 0.95rem; opacity: 0.9;">🚀 Powered by T5 AI | 💡 Expert Financial Guidance | ⚡ Real-time Answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("💬 Chats")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
                create_new_chat()
                st.rerun()
        with col2:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_chat_btn", 
                        disabled=len(st.session_state.chat_history) == 0,
                        help="Clear current chat history"):
                clear_history()
                st.rerun()
        
        st.divider()
        
        if st.session_state.all_chats:
            
            for chat_session in reversed(st.session_state.all_chats):
                chat_id = chat_session['id']
                chat_title = chat_session['title']
                last_updated = chat_session['last_updated']
                
                is_active = (chat_id == st.session_state.current_chat_id)
                
                if is_active:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 0.75rem; border-radius: 8px; margin-bottom: 0.5rem; 
                                border-left: 4px solid #fff; cursor: pointer;">
                        <div style="font-size: 0.9rem; color: white; font-weight: 600; margin-bottom: 0.25rem;">
                            💬 {chat_title}
                        </div>
                        <div style="font-size: 0.7rem; color: rgba(255,255,255,0.8);">
                        {last_updated.split()[1] if ' ' in last_updated else last_updated}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Delete button for active chat
                    if len(st.session_state.all_chats) > 1:  # Don't allow deleting if it's the only chat
                        if st.button("🗑️ Delete this chat", key=f"delete_active_{chat_id}", use_container_width=True):
                            delete_chat(chat_id)
                            st.rerun()
                else:
                    # Use columns for inactive chat: chat button + delete button
                    col_chat, col_del = st.columns([4, 1])
                    with col_chat:
                        if st.button(
                            f"💬 {chat_title}\n{last_updated.split()[1] if ' ' in last_updated else last_updated}",
                            key=f"chat_{chat_id}",
                            use_container_width=True
                        ):
                            load_chat(chat_id)
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True, help="Delete chat"):
                            delete_chat(chat_id)
                            st.rerun()
                        
            # Export current conversation
            if st.session_state.chat_history:
                conversation_json = export_conversation()
                if conversation_json:
                    st.download_button(
                        label="📥 Export Current Chat",
                        data=conversation_json,
                        file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        else:
            st.caption("No chats yet. Click '➕ New Chat' to start!")
        
        st.divider()
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            # User message
            st.markdown(f"""
            <div class="message-row user">
                <div class="message-avatar user-avatar">👤</div>
                <div class="message-content">
                    <div class="user-message">
                        {chat['question']}
                    </div>
                    <div style="text-align: right; margin-top: 0.25rem;">
                        <small style="color: #999; font-size: 0.75rem;">{chat['timestamp']}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            answer_text = chat['answer']
            
            if answer_text == "typing...":
                st.markdown(f"""
                <div class="message-row bot">
                    <div class="message-avatar bot-avatar">🤖</div>
                    <div class="message-content">
                        <div class="bot-message">
                            <div style="display: flex; gap: 0.5rem; align-items: center;">
                                <span>Thinking</span>
                                <div style="display: flex; gap: 0.25rem;">
                                    <div style="width: 6px; height: 6px; border-radius: 50%; background: #667eea; animation: typing 1.4s infinite;"></div>
                                    <div style="width: 6px; height: 6px; border-radius: 50%; background: #667eea; animation: typing 1.4s infinite 0.2s;"></div>
                                    <div style="width: 6px; height: 6px; border-radius: 50%; background: #667eea; animation: typing 1.4s infinite 0.4s;"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <style>
                @keyframes typing {{
                    0%, 60%, 100% {{ opacity: 0.3; transform: translateY(0); }}
                    30% {{ opacity: 1; transform: translateY(-4px); }}
                }}
                </style>
                """, unsafe_allow_html=True)
                continue
            
            response_time = chat.get('response_time', 0)
            
            st.markdown(f"""
            <div class="message-row bot">
                <div class="message-avatar bot-avatar">🤖</div>
                <div class="message-content">
                    <div class="bot-message">
                        {answer_text}
                    </div>
                    <div style="margin-top: 0.25rem;">
                        <small style="color: #999; font-size: 0.75rem;">
                            ⏱️ {response_time:.2f}s
                        </small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <p style="font-size: 1.1rem; color: #666; margin-bottom: 1rem;">
                Click on any FAQ below or type your question to begin!
            </p>
            <p style="color: #999; font-size: 0.9rem;">
                💡 Your conversation will appear here
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('''
    </div>
    <script>
        const chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
    ''', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <style>
    div[data-testid="stForm"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 2px solid #e5e7eb;
    }
    
    div[data-testid="stForm"] input {
        font-size: 1rem !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
    }
    
    div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    is_generating = (st.session_state.chat_history and 
                     st.session_state.chat_history[-1]['answer'] == "typing...")
    
    if not st.session_state.sample_questions_used and len(st.session_state.chat_history) == 0:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <p style="font-size: 0.95rem; color: #666; font-weight: 500;">
                ✨ Get started with these sample questions:
            </p>
        </div>
        <style>
        div[data-testid="column"] button {
            height: 70px !important;
            white-space: normal !important;
            font-size: 0.85rem !important;
            line-height: 1.3 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        starter_questions = [
            {"emoji": "📈", "question": "Stock vs Bond"},
            {"emoji": "💰", "question": "Compound Interest"},
            {"emoji": "🏦", "question": "What is RRSP?"},
            {"emoji": "📊", "question": "Diversification"},
            {"emoji": "💳", "question": "Capital Gains Tax"},
        ]
        
        full_questions = [
            "What is the difference between a stock and a bond?",
            "How does compound interest work?",
            "What is an RRSP?",
            "Explain diversification in investing",
            "How do I calculate capital gains tax?",
        ]
        
        cols = st.columns(len(starter_questions))
        for idx, (col, faq, full_q) in enumerate(zip(cols, starter_questions, full_questions)):
            with col:
                if st.button(
                    f"{faq['emoji']}\n{faq['question']}", 
                    key=f"starter_{idx}",
                    use_container_width=True,
                    help=full_q
                ):
                    # Immediately flag samples as used to prevent re-render in this run
                    st.session_state.sample_questions_used = True
                    is_valid, error_msg = validate_question(full_q)
                    if is_valid and not is_duplicate_pending(full_q):
                        add_to_history(full_q, "typing...", 0, 0)
                    # Rerun immediately to prevent further rendering in this cycle
                    st.rerun()
                    break  # Exit loop after rerun (defensive, though rerun stops execution)
        
        st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    
    if not is_generating:
        with st.form(key="question_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                question = st.text_input(
                    "Message FinanceGPT...",
                    placeholder="💬 Ask me anything about finance, investments, banking, taxes...",
                    label_visibility="collapsed",
                    key="question_input"
                )
            with col2:
                submit_button = st.form_submit_button("Send 📤", use_container_width=True, type="primary")
        
        if submit_button and question:
            is_valid, error_msg = validate_question(question)
            
            if not is_valid:
                st.error(f"❌ {error_msg}")
            else:
                # Prevent duplicate pending entries if user submits while a matching request is in progress
                if not is_duplicate_pending(question):
                    add_to_history(question, "typing...", 0, 0)
                st.rerun()
    
    if (st.session_state.chat_history and 
        st.session_state.chat_history[-1]['answer'] == "typing..."):
        
        latest_chat = st.session_state.chat_history[-1]
        question = latest_chat['question']
        
        answer, response_time, confidence = generate_answer(
            question,
            st.session_state.model,
            st.session_state.tokenizer,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature
        )
        
        st.session_state.chat_history[-1]['answer'] = answer
        st.session_state.chat_history[-1]['response_time'] = response_time
        st.session_state.chat_history[-1]['confidence'] = confidence
        st.session_state.total_response_time += response_time
        
        st.rerun()
    
    with st.expander("💭 More Sample Questions You Can Ask"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Personal Finance:**
            - What is a credit score?
            - How does a mortgage work?
            - What are the benefits of a high-yield savings account?
            - How do I create a budget?
            
            **Investing:**
            - What is dollar-cost averaging?
            - How do mutual funds work?
            - What are dividend stocks?
            """)
        with col2:
            st.markdown("""
            **Retirement:**
            - What's the difference between RRSP and TFSA?
            - When should I start saving for retirement?
            - What is a pension plan?
            
            **Business & Economics:**
            - What causes inflation?
            - How does the stock market affect the economy?
            - What is GDP?
            """)
    
    st.divider()
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                text-align: center; 
                margin-top: 2rem;">
        <h3 style="color: #667eea; margin-bottom: 1rem;">About FinanceGPT</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 2rem;">🤖</div>
                <strong>AI-Powered Advisory</strong><br>
                <span style="color: #666; font-size: 0.9rem;">T5 Transfer Learning</span>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 2rem;">📚</div>
                <strong>49,000+ Financial Q&A Pairs</strong><br>
                <span style="color: #666; font-size: 0.9rem;">Expert Training Data</span>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 2rem;">💡</div>
                <strong>Financial Guidance</strong><br>
                <span style="color: #666; font-size: 0.9rem;">Personal Finance Advice</span>
            </div>
            <div style="flex: 1; min-width: 200px;">
                <div style="font-size: 2rem;">🔒</div>
                <strong>Secure & Private</strong><br>
                <span style="color: #666; font-size: 0.9rem;">Your Data Stays Safe</span>
            </div>
        </div>

    </div>
    """.format(datetime.now().strftime('%B %d, %Y at %H:%M')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
