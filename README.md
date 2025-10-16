# 💰 FinanceGPT - AI Financial Advisory Chatbot

An intelligent conversational AI system fine-tuned on financial domain knowledge to provide expert-level guidance on personal finance, investments, banking, taxation, and more.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)

## 🎯 Overview

FinanceGPT is a domain-specialized chatbot built by fine-tuning Google's T5 (Text-to-Text Transfer Transformer) model on a comprehensive dataset of ~49,000 financial question-answer pairs. The system provides accurate, accessible financial advice through an intuitive conversational interface.

### Key Features

- 🤖 **AI-Powered**: Built on T5-Base with 220M parameters, fine-tuned for financial expertise
- 📚 **Comprehensive Knowledge**: Trained on 49,000+ Q&A pairs from Investopedia and curated financial datasets
- 💡 **Expert Guidance**: Provides advisor-level answers on investments, taxes, banking, retirement, and more
- ⚡ **Real-time Responses**: Optimized inference with 3-6 second response times
- 💬 **Multi-Session Chats**: Manage multiple conversations with history and export capabilities
- 🎯 **Confidence Scoring**: Built-in confidence estimation with appropriate disclaimers
- 🔒 **Secure & Private**: All processing happens locally, your data stays safe

## 📋 Prerequisites

Before running the application, ensure you have:

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **8GB+ RAM** (16GB recommended for optimal performance)
- **GPU** (Optional but recommended for faster inference)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/IrakizaGaius/FinancialChatbot.git
cd FinancialChatbot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**

```txt
streamlit>=1.20.0
tensorflow>=2.10.0
transformers>=4.25.0
pandas>=1.5.0
```

### 4. Download the Model

**Pre-trained Model (Recommended)**

Download the fine-tuned model from [model hosting location] and place it in:

```
Chatbot/
├── T5_finetuned_model/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── spiece.model
```

### 5. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 💻 Usage

### Starting a Conversation

1. **Launch the app** using `streamlit run streamlit_app.py`
2. **Click on sample questions** or type your own financial question
3. **Wait for the AI response** (3-6 seconds with typing animation)
4. **Continue the conversation** or start a new chat

### Sample Questions

- "What is the difference between a stock and a bond?"
- "How does compound interest work?"
- "What is an RRSP?"
- "Explain diversification in investing"
- "How do I calculate capital gains tax?"

### Managing Conversations

- **➕ New Chat**: Start a fresh conversation
- **🗑️ Clear**: Clear current chat history
- **💬 Chat List**: Switch between previous conversations
- **📥 Export**: Download conversation as JSON
- **🗑️ Delete**: Remove specific chats

## 🏗️ Project Structure

```
Chatbot/
├── streamlit_app.py              # Main Streamlit application
├── T5_base.py                    # Model training script
├── scrapeInvestopedia.py         # Data scraping script
├── Financial_Dataset_Preprocessing.ipynb  # Data preprocessing
├── T5_finetuned_model/           # Fine-tuned model (excluded from git)
├── Financial dataset/            # Training data (excluded from git)
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🔧 Configuration

### GPU Configuration

For GPU acceleration (recommended):

```python
# Automatic GPU detection and configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Generation Parameters

Customize response generation in `streamlit_app.py`:

```python
max_length = 1024        # Maximum answer length
num_beams = 1            # Beam search (1 = sampling mode)
temperature = 0.7        # Sampling temperature (0.0-1.0)
top_p = 0.95            # Nucleus sampling threshold
```

## 📊 Model Performance

- **Training Dataset**: ~49,000 financial Q&A pairs
- **Validation Split**: 90/10 train-validation
- **Average Response Time**: 4.2 seconds
- **Average Confidence Score**: 0.64
- **ROUGE-L F1**: 0.72 (validation set)

## 🛠️ Troubleshooting

### Model Loading Issues

**Error**: `Failed to load model`

**Solution**:

```bash
# Verify model directory exists
ls T5_finetuned_model/

# Check file permissions
chmod -R 755 T5_finetuned_model/

# Try alternative loading by setting in code:
# from_pt=True  # Convert from PyTorch
```

### Memory Issues

**Error**: `OOM (Out of Memory)`

**Solution**:

- Reduce `max_length` to 512 or 768
- Close other applications
- Use CPU instead of GPU (slower but uses system RAM)
- Increase swap space (Linux/macOS)

### Slow Response Times

**Solution**:

- Ensure GPU is being used (check with `nvidia-smi`)
- Reduce `max_length` parameter
- Use `num_beams=1` for sampling (already default)
- Close unnecessary browser tabs

### Port Already in Use

**Error**: `Port 8501 is already in use`

**Solution**:

```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9
```

## 🧪 Testing

Test the installation:

```bash
# Test model loading
python -c "from transformers import T5Tokenizer; T5Tokenizer.from_pretrained('t5-base')"

# Test TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Test Streamlit
streamlit hello
```

## 📚 Additional Resources

- **Live Demo**: https://huggingface.co/spaces/GaiusIrakiza/FinanceGpt
- **GitHub Repository**: https://github.com/IrakizaGaius/FinancialChatbot
- **Kaggle Notebook**: https://www.kaggle.com/code/gaiusirakiza/financial-dataset-transfer-learning
- **Model Repository**: https://huggingface.co/GaiusIrakiza/financegpt-t5-model
- **Technical Report**: Comprehensive academic report on the project methodology
- **Training Notebook**: `Financial_Dataset_Preprocessing.ipynb` - Data preprocessing pipeline
- **Model Training**: `T5_base.py` - Fine-tuning script with hyperparameters

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Retrieval-Augmented Generation (RAG)**: Integrate vector database for real-time knowledge
2. **Multi-turn Conversations**: Add context tracking across questions
3. **Personalization**: User profiles and preference learning
4. **Evaluation Framework**: Automated metrics and benchmarking
5. **Additional Languages**: Multilingual support

## ⚠️ Disclaimer

**Important**: FinanceGPT is designed for educational purposes and general financial information. It is NOT a substitute for professional financial advice. Always consult with qualified financial advisors, accountants, or legal professionals before making important financial decisions.

The AI model may occasionally produce incorrect or incomplete information. Users should verify critical information through authoritative sources.
