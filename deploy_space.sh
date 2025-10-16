#!/bin/bash

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

if [ "$#" -ne 1 ]; then
    echo -e "${RED}Usage: $0 SPACE_NAME${NC}"
    echo "Example: $0 financegpt"
    exit 1
fi

SPACE_NAME=$1
USERNAME="GaiusIrakiza"
SPACE_URL="https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}"

echo -e "${BLUE}🚀 Deploying FinanceGPT to Hugging Face Spaces${NC}"
echo -e "${BLUE}Space URL: ${SPACE_URL}${NC}"
echo ""

# Create temp directory
TEMP_DIR="/tmp/hf_space_deploy"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo -e "${GREEN}Step 1: Cloning Space repository...${NC}"
git clone "https://huggingface.co/spaces/${USERNAME}/${SPACE_NAME}" "$TEMP_DIR" || {
    echo -e "${RED}Error: Could not clone Space${NC}"
    echo "Make sure the Space exists at: ${SPACE_URL}"
    exit 1
}

cd "$TEMP_DIR"

echo -e "${GREEN}Step 2: Copying files...${NC}"
cp /Users/irakizagaius/Downloads/Chatbot/streamlit_app.py .
cp /Users/irakizagaius/Downloads/Chatbot/requirements.txt .

echo -e "${GREEN}Step 3: Creating README.md...${NC}"
cat > README.md << 'EOF'
---
title: FinanceGPT - AI Financial Advisor
emoji: 💰
colorFrom: purple
colorTo: pink
sdk: streamlit
sdk_version: "1.50.0"
app_file: streamlit_app.py
pinned: false
license: mit
---

# 💰 FinanceGPT - AI Financial Advisor

An intelligent financial advisory chatbot powered by fine-tuned T5 AI, trained on 49,000+ expert financial Q&A pairs from Investopedia and curated financial datasets.

## 🚀 Features

- 🤖 **Expert Financial Knowledge**: Covers investments, banking, taxes, credit, retirement planning
- ⚡ **Real-time Answers**: Get instant responses to financial questions (3-6 seconds)
- 💬 **Multi-Chat Sessions**: Manage multiple conversation threads with history
- 🎯 **Confidence Scoring**: Built-in answer reliability assessment
- 📥 **Export Conversations**: Download your chat history as JSON

## 💡 How to Use

1. Type your financial question in the chat box
2. Press Enter or click Send
3. Get expert AI-powered guidance instantly

Try asking:
- "What is the difference between a stock and a bond?"
- "How does compound interest work?"
- "What is an RRSP?"
- "Explain diversification in investing"

## 📊 Model Details

- **Base Model**: T5-Base (220M parameters)
- **Training Data**: 49,000+ financial Q&A pairs
- **Evaluation**: ROUGE-L F1 score of 0.72
- **Average Response Time**: 4.2 seconds

## ⚠️ Disclaimer

**Important**: This AI chatbot provides educational information only and should NOT be considered professional financial advice. Always consult with qualified financial advisors, accountants, or legal professionals before making important financial decisions.

The AI model may occasionally produce incorrect or incomplete information. Users should verify critical information through authoritative sources.

---

**Model Repository**: [GaiusIrakiza/financegpt-t5-model](https://huggingface.co/GaiusIrakiza/financegpt-t5-model)  
**GitHub Repository**: [IrakizaGaius/FinancialChatbot](https://github.com/IrakizaGaius/FinancialChatbot)  
**Kaggle Notebook**: [Financial Dataset Transfer Learning](https://www.kaggle.com/code/gaiusirakiza/financial-dataset-transfer-learning)

Made with ❤️ for financial education and empowerment
EOF

echo -e "${GREEN}Step 4: Committing changes...${NC}"
git add .
git commit -m "Deploy FinanceGPT - AI Financial Advisory Chatbot

- Add fine-tuned T5 model for financial question answering
- Add Streamlit web interface with multi-chat support
- Trained on 49,000+ financial Q&A pairs
- Includes confidence scoring and export functionality
"

echo -e "${GREEN}Step 5: Pushing to Hugging Face...${NC}"
git push || {
    echo -e "${RED}Error: Push failed${NC}"
    echo "You may need to authenticate. Run:"
    echo "  huggingface-cli login"
    exit 1
}

echo ""
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo ""
echo -e "${BLUE}Your Space is building at:${NC}"
echo -e "${BLUE}${SPACE_URL}${NC}"
echo ""
echo "Monitor the build logs in the 'Logs' tab."
echo "Your app will be live in 2-5 minutes!"
echo ""

# Cleanup
cd /
rm -rf "$TEMP_DIR"
