# config.py-configuration file 
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve() #used for internal default paths 
# points to project directory 'Multimodal_project' in my case 

# PDF to index (change to your PDF path)
PDF_FILEPATH = "/home/students/yadav/workspace/Multimodal_project/1706.03762v7.pdf"
#CHROMA_PATH = str(ROOT / "chroma_store")

# Chroma DB persistence directory
CHROMA_PATH = str(ROOT / "chroma_store") # path to where vector DB is stored 

# Models (local/huggingface)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_SUMMARIZER = "google/flan-t5-small"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"  # BLIP-large- image captioning model
QA_MODEL = "deepset/roberta-base-squad2"  #  model used for extractive QA
GENERATIVE_FALLBACK = "google/flan-t5-base"  # optional, small gen model (in case QA model fails)

# Indexing options
TOP_K = 4 #retrieve 4 relevant chunks from vector DB, can be changes 
#if we want to somehow use this ablations studies or something.
EMBED_BATCH = 64 #batch size used when computing embeddings 
#for the embedding for the summaries

# Summarizer chunk sizes (adjust if needed, Internal limits for the summarizer )
SUMMARY_MAX_TOKENS = 150 
SUMMARY_MIN_TOKENS = 30
