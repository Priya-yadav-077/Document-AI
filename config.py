# config.py - configuration file 
import os
from pathlib import Path
import glob

# Get the project root directory
ROOT = Path(__file__).parent.resolve()

def find_pdf_file():
    """Automatically find PDF file in project directory or subdirectories."""
    # Check specific common locations first
    common_paths = [
        ROOT / "ResearchPaper",
        ROOT / "papers",
        ROOT / "pdfs",
        ROOT,
    ]
    
    # Search each location
    for path in common_paths:
        if path.exists():
            pdf_files = list(path.glob("*.pdf"))
            if pdf_files:
                return str(pdf_files[0])
    
    # If not found, do recursive search
    pdf_files = list(ROOT.glob("**/*.pdf"))
    if pdf_files:
        return str(pdf_files[0])
    
    return None

# PDF to index (auto-detected or set manually)
PDF_FILEPATH = find_pdf_file()

# Chroma DB persistence directory (stored in project root)
CHROMA_PATH = str(ROOT / "chroma_store")

# Models (HuggingFace - works on any platform)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_SUMMARIZER = "google/flan-t5-small"
IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
QA_MODEL = "deepset/roberta-base-squad2"

# Retrieval options
TOP_K = 4
INITIAL_RETRIEVAL_K = 20  # Retrieve more chunks initially for filtering (increased for better recall)
EMBED_BATCH = 64

# Relevance filtering settings
ENABLE_RELEVANCE_FILTERING = True
SIMILARITY_THRESHOLD = 0.2  # Stage 1: Minimum cosine similarity score (lowered for better recall)
ENABLE_RERANKING = True  # Stage 2: Use cross-encoder reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_AFTER_RERANK = 6  # Final number of chunks to use (increased for more context)
RERANKER_TOP_K = 6  # Alias for evaluation module compatibility

# Answer generation settings  
QA_MODE = "generative"  # Options: "extractive", "generative", "hybrid"
GENERATIVE_QA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"  # Llama 3.2 for true synthesis
ANSWER_MAX_LENGTH = 250  # Max tokens to generate (increased to allow complete, detailed answers)
ANSWER_MIN_LENGTH = 100  # Min tokens to generate (ensures comprehensive answers matching QASPER reference length)
CONTEXT_MAX_CHARS = 3000 # Max chars of context to use (doubled for more information)

# Summarizer chunk sizes
SUMMARY_MAX_TOKENS = 150 
SUMMARY_MIN_TOKENS = 30

# Multi-PDF comparison settings
MULTI_PDF_MODE = True
COMPARISON_METHOD = "llama_judge"  # Options: "llama_judge", "similarity", "hybrid"
JUDGE_MAX_LENGTH = 300  # Max tokens for judge decision
MULTI_CHROMA_PATH = str(ROOT / "chroma_multi_store")  # Separate store for multi-PDF mode
