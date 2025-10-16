from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import shutil

# ============================================
# CONFIG
# ============================================
MODEL_NAME = "t5-base"
SAVE_DIR = "t5_base_local"
ZIP_PATH = "t5_base_local.zip"

# ============================================
# DOWNLOAD MODEL & TOKENIZER
# ============================================
print(f"⬇️ Downloading {MODEL_NAME} ...")

# Load model in half precision to save RAM during download
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Save locally
os.makedirs(SAVE_DIR, exist_ok=True)
tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"✅ Model and tokenizer saved to: {SAVE_DIR}")

# ============================================
# COMPRESS INTO ZIP
# ============================================
print("🗜️ Compressing into zip file...")
shutil.make_archive(SAVE_DIR, 'zip', SAVE_DIR)
print(f"✅ Compressed model saved as: {ZIP_PATH}")