# config.py
import os

# Model Ayarları
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Dizin Ayarları
DB_DIR = "./chroma_db"

# RAG Ayarları
RETRIEVER_K = 2 # Aramada getirilecek parça sayısı