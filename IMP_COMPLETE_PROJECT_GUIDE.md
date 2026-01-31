# DOCUMENT-AI: COMPLETE PROJECT IMPLEMENTATION GUIDE

**Version:** 2.0  
**Last Updated:** January 2026  
**Purpose:** Comprehensive guide to understand and explain the entire RAG system

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Core Concepts](#2-core-concepts)
3. [System Architecture](#3-system-architecture)
4. [File-by-File Implementation](#4-file-by-file-implementation)
5. [The RAG Pipeline Explained](#5-the-rag-pipeline-explained)
6. [Multi-PDF Comparison](#6-multi-pdf-comparison)
7. [QASPER Evaluation System](#7-qasper-evaluation-system)
8. [Performance Optimization Journey](#8-performance-optimization-journey)
9. [How to Use (Complete Guide)](#9-how-to-use-complete-guide)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. PROJECT OVERVIEW

### What Is This Project?

**Document-AI** is an intelligent research paper analysis system that:
- Reads PDF research papers
- Understands their content (text, tables, images)
- Answers questions about them using AI
- Can compare answers from multiple papers
- Evaluates its performance on scientific benchmarks

### Real-World Example

```
INPUT:
- PDF: "Deep Learning for Computer Vision" (research paper)
- Question: "What datasets were used for evaluation?"

SYSTEM PROCESS:
1. Extracts all content from PDF
2. Breaks it into searchable chunks
3. Finds chunks related to "evaluation datasets"
4. Reads those chunks carefully
5. Generates intelligent answer

OUTPUT:
"The paper evaluated the model on ImageNet (1.2M images, 1000 classes), 
COCO (330K images for object detection), and a custom dataset of 50K 
medical images. ImageNet was used for classification tasks while COCO 
was used for detection and segmentation."
```

### Why This Matters

**Before this system:**
- Manual reading of papers (slow)
- Hard to compare multiple papers
- No way to measure answer quality objectively

**With this system:**
- Instant answers from any paper
- Compare answers across papers automatically
- Scientific metrics (F1, ROUGE) to measure performance
- Scalable to hundreds of papers

---

## 2. CORE CONCEPTS

### 2.1 What is RAG (Retrieval-Augmented Generation)?

RAG combines two AI capabilities:

**Retrieval:**
- Search for relevant information (like Google search)
- Find the most relevant parts of documents
- Uses similarity between question and document chunks

**Generation:**
- Create human-like text (like ChatGPT)
- Synthesize information into coherent answers
- Uses large language models (LLMs)

**Why combine them?**
- Pure search: Finds relevant text but doesn't synthesize
- Pure generation: Can hallucinate (make up facts)
- RAG: Grounds generation in retrieved facts (accurate + fluent)

### 2.2 Key Technologies

#### Vector Embeddings
```
Text → Numbers that capture meaning

Example:
"machine learning" → [0.2, -0.5, 0.8, ..., 0.3] (384 numbers)
"artificial intelligence" → [0.19, -0.48, 0.79, ..., 0.31] (similar!)
"banana recipe" → [-0.6, 0.2, -0.1, ..., 0.7] (very different!)

Purpose: Find similar concepts even with different words
```

#### Vector Database (ChromaDB)
- Stores embeddings and original text
- Searches by similarity (not just keyword matching)
- Fast: Can search millions of chunks in milliseconds

#### Large Language Models (LLMs)
- Llama-3.2-3B-Instruct: Generates comprehensive answers
- RoBERTa-SQuAD2: Extracts exact answer spans
- FLAN-T5-small: Summarizes long text
- BLIP: Describes images

### 2.3 Two-Stage Filtering

Traditional RAG problem: Too much irrelevant context

**Our Solution:**

**Stage 1: Similarity Filtering**
```
Retrieved: 20 chunks
↓
Calculate cosine similarity with question
↓
Keep chunks with similarity > threshold (0.2)
↓
Result: ~10-15 chunks
```

**Stage 2: Cross-Encoder Reranking**
```
Input: ~10-15 filtered chunks
↓
Cross-encoder scores each chunk for relevance
(More accurate than embeddings but slower)
↓
Sort by score, keep top 6
↓
Result: 6 most relevant chunks
```

**Why two stages?**
- Stage 1: Fast, removes obviously irrelevant chunks
- Stage 2: Accurate, ranks remaining chunks precisely

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DOCUMENT-AI SYSTEM                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   PDF Input  │────│   Loader     │────│  Extracted   │ │
│  │              │    │ (Unstructured)│    │   Content    │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│                                                   │          │
│                                                   ▼          │
│                                          ┌─────────────────┐│
│                                          │  Preprocessor   ││
│                                          │  - Chunk text   ││
│                                          │  - Summarize    ││
│                                          │  - Caption imgs ││
│                                          └────────┬────────┘│
│                                                   │          │
│                                                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Embeddings  │────│  ChromaDB    │────│   Indexed    │ │
│  │(Sent. Trans.)│    │(Vector Store)│    │   Chunks     │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
│  ════════════════════ INDEXING COMPLETE ═══════════════════ │
│                                                              │
│  ┌──────────────┐                                           │
│  │   Question   │                                           │
│  └──────┬───────┘                                           │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Retrieve   │────│ Stage 1 Filter│───│ Stage 2 Rerank│ │
│  │  (Semantic)  │    │  (Similarity) │    │(Cross-Encoder)│ │
│  └──────────────┘    └──────────────┘    └──────┬────────┘ │
│                                                   │          │
│                                                   ▼          │
│                                          ┌─────────────────┐│
│                                          │  Top 6 Chunks   ││
│                                          └────────┬────────┘│
│                                                   │          │
│                                                   ▼          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Extractive  │────│  Quality     │────│  Generative  │ │
│  │QA (RoBERTa)  │    │  Check       │    │ QA (Llama)   │ │
│  │              │    │ (Confidence) │    │              │ │
│  └──────────────┘    └──────────────┘    └──────┬───────┘ │
│         │                                         │          │
│         └──────────────┬──────────────────────────┘          │
│                        ▼                                     │
│                 ┌──────────────┐                            │
│                 │ Final Answer │                            │
│                 └──────────────┘                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

**Indexing Phase:**
```
PDF → [Extract] → Raw Content → [Chunk] → Text Chunks
                                             ↓
Text Chunks → [Embed] → Vector Embeddings → [Store] → ChromaDB
                                                         ↓
                                                    Searchable Index
```

**Query Phase:**
```
Question → [Embed] → Question Vector
                          ↓
Question Vector → [Search ChromaDB] → Top 20 Chunks
                                           ↓
Top 20 Chunks → [Filter by Similarity] → ~12 Chunks
                                           ↓
~12 Chunks → [Rerank with Cross-Encoder] → Top 6 Chunks
                                              ↓
Top 6 Chunks → [Try Extractive QA] → Good Answer?
                                           ↓
                                      Yes → Return
                                      No  → [Generative QA] → Final Answer
```

---

## 4. FILE-BY-FILE IMPLEMENTATION

### 4.1 Configuration (`config.py`)

**Purpose:** Central configuration for all system settings

**Key Sections:**

```python
# ===== MODELS =====
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Converts text to 384-dimensional vectors
# Fast (15ms per chunk) and accurate enough

GENERATIVE_QA_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
# 3 billion parameter model for answer generation
# Requires ~7GB GPU memory with quantization

QA_MODEL = "deepset/roberta-base-squad2"
# Extractive QA: finds exact answer spans in text
# Trained on SQuAD2 dataset (100K questions)

# ===== RETRIEVAL SETTINGS =====
INITIAL_RETRIEVAL_K = 20
# How many chunks to retrieve initially
# Higher = better recall, but slower reranking

SIMILARITY_THRESHOLD = 0.2
# Minimum cosine similarity to keep chunk
# Range: 0.0 (unrelated) to 1.0 (identical)
# 0.2 = permissive (keep more chunks)

RERANKER_TOP_K = 6
# Final number of chunks after reranking
# More chunks = more context but more tokens

# ===== ANSWER GENERATION =====
ANSWER_MAX_LENGTH = 250
# Maximum tokens to generate
# QASPER references average 119 tokens

ANSWER_MIN_LENGTH = 100
# Minimum tokens to generate
# Ensures complete answers

CONTEXT_MAX_CHARS = 3000
# Maximum characters of context for LLM
# Llama-3.2 supports up to 8K tokens
```

**Why These Values?**

| Setting | Value | Reasoning |
|---------|-------|-----------|
| INITIAL_RETRIEVAL_K | 20 | Balances recall (finding relevant chunks) vs speed |
| SIMILARITY_THRESHOLD | 0.2 | Permissive enough to not miss relevant content |
| RERANKER_TOP_K | 6 | Provides enough context without exceeding token limits |
| ANSWER_MAX_LENGTH | 250 | Allows comprehensive answers matching reference length |

### 4.2 PDF Loader (`loader.py`)

**Purpose:** Extract all content (text, tables, images) from PDFs

**Implementation:**

```python
from unstructured.partition.pdf import partition_pdf

def load_pdf_elements(pdf_path):
    """
    Extract structured elements from PDF
    
    Process:
    1. Use Unstructured library to parse PDF
    2. Detect element types (text, table, image)
    3. Extract each element with metadata
    4. Return structured list
    """
    
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",  # High-resolution extraction (slower but accurate)
        infer_table_structure=True,  # Parse tables into structured format
        extract_images_in_pdf=True,  # Extract embedded images
        extract_image_block_types=["Image", "Figure"]
    )
    
    chunks = []
    for elem in elements:
        if elem.category == "Table":
            # Tables converted to markdown format
            chunks.append({
                'type': 'table',
                'content': elem.metadata.text_as_html,  # HTML table
                'metadata': {'page': elem.metadata.page_number}
            })
        elif elem.category in ["Image", "Figure"]:
            # Images stored as base64
            chunks.append({
                'type': 'image',
                'content': elem.metadata.image_base64,
                'metadata': {'page': elem.metadata.page_number}
            })
        else:
            # Text paragraphs
            chunks.append({
                'type': 'text',
                'content': elem.text,
                'metadata': {
                    'page': elem.metadata.page_number,
                    'category': elem.category
                }
            })
    
    return chunks
```

**Why Unstructured?**
- Handles complex PDF layouts (multi-column, embedded objects)
- Better table extraction than PyPDF2
- Preserves document structure

### 4.3 Summarizer (`summarizer.py`)

**Purpose:** Summarize long text chunks to improve embedding quality

**Implementation:**

```python
from transformers import pipeline

_summarizer = None

def get_summarizer():
    """Lazy initialization of summarizer"""
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="google/flan-t5-small",  # 80M parameters
            device=0 if torch.cuda.is_available() else -1
        )
    return _summarizer

def summarize_text(text, max_length=150, min_length=30):
    """
    Summarize text for better embeddings
    
    When to use:
    - Text > 512 tokens (embedding model limit)
    - Verbose text with lots of filler
    
    Process:
    1. Truncate to model limit (512 tokens)
    2. Generate summary using FLAN-T5
    3. Use summary for embedding (original stored for retrieval)
    """
    summarizer = get_summarizer()
    
    if len(text.split()) < 100:
        return text  # Don't summarize short text
    
    result = summarizer(
        text[:2048],  # FLAN-T5 input limit
        max_length=max_length,
        min_length=min_length,
        do_sample=False  # Deterministic (same input = same output)
    )
    
    return result[0]['summary_text']
```

**When is this used?**
- During indexing for very long chunks
- Improves embedding quality by focusing on main points
- Original text still stored and used for answer generation

### 4.4 Vision Module (`vision.py`)

**Purpose:** Generate text descriptions of images

**Implementation:**

```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64

_vision_model = None
_vision_processor = None

def get_vision_model():
    """Load BLIP model for image captioning"""
    global _vision_model, _vision_processor
    if _vision_model is None:
        _vision_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        _vision_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    return _vision_model, _vision_processor

def summarize_image(image_base64):
    """
    Generate text description of image
    
    Process:
    1. Decode base64 to image
    2. Run BLIP model to generate caption
    3. Return caption for indexing
    
    Example:
    Image → "a diagram showing the neural network architecture
             with three convolutional layers followed by pooling"
    """
    model, processor = get_vision_model()
    
    # Decode base64 to PIL Image
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))
    
    # Generate caption
    inputs = processor(image, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=100)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    return caption
```

**Why caption images?**
- Makes images searchable by text query
- User asks: "What does the architecture look like?"
- System finds image captions mentioning "architecture"

### 4.5 RAG Pipeline (`rag_pipeline.py`)

**Purpose:** Core RAG logic for single PDF

**Key Functions:**

#### 4.5.1 Setup and Indexing

```python
def setup_retriever(pdf_path: str, use_summarization: bool = False):
    """
    Index a PDF into ChromaDB
    
    Steps:
    1. Extract content from PDF
    2. Process each chunk (summarize if needed, caption images)
    3. Generate embeddings
    4. Store in ChromaDB with metadata
    5. Save docstore (maps IDs to original content)
    """
    
    # Extract PDF content
    chunks = load_pdf_elements(pdf_path)
    
    # Initialize ChromaDB
    collection = init_chroma()
    embedder = get_embedder()
    
    docstore = {}
    
    for idx, chunk in enumerate(chunks):
        chunk_id = f"chunk_{idx}"
        
        # Process based on chunk type
        if chunk['type'] == 'text':
            content = chunk['content']
            
            # Optional: Summarize long text
            if use_summarization and len(content) > 500:
                summarizer = get_summarizer()
                summary = summarize_text(content)
                embed_text = f"summary: {summary}"
            else:
                embed_text = content[:512]  # Truncate to embedding limit
            
        elif chunk['type'] == 'table':
            # Tables: use markdown representation
            content = chunk['content']
            embed_text = f"table: {content[:512]}"
            
        elif chunk['type'] == 'image':
            # Images: caption and store base64
            caption = summarize_image(chunk['content'])
            embed_text = f"image: {caption}"
            content = caption  # Use caption for retrieval
        
        # Generate embedding
        embedding = embedder.encode(embed_text, convert_to_tensor=False)
        
        # Store in ChromaDB
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[content],
            metadatas=[chunk['metadata']]
        )
        
        # Store original content in docstore
        docstore[chunk_id] = chunk
    
    # Save docstore to disk
    save_docstore(docstore)
    
    return collection
```

#### 4.5.2 Two-Stage Filtering

```python
def apply_relevance_filtering(
    question: str,
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    distances: List[float]
) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Two-stage filtering for better context selection
    
    Stage 1: Similarity Threshold
    - Convert distance to similarity
    - Keep chunks above threshold
    - Fast filtering (no model inference)
    
    Stage 2: Cross-Encoder Reranking
    - Score each chunk with cross-encoder
    - More accurate than embeddings
    - Select top-K chunks
    """
    
    # STAGE 1: Similarity filtering
    filtered = []
    for doc, meta, doc_id, distance in zip(documents, metadatas, ids, distances):
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
        
        if similarity >= SIMILARITY_THRESHOLD:
            filtered.append((doc, meta, doc_id, similarity))
    
    if not filtered:
        # No chunks passed, use top results anyway
        return documents[:TOP_K], metadatas[:TOP_K], ids[:TOP_K]
    
    # STAGE 2: Cross-encoder reranking
    if not ENABLE_RERANKING:
        # Skip reranking, just return filtered results
        return (
            [x[0] for x in filtered[:TOP_K_AFTER_RERANK]],
            [x[1] for x in filtered[:TOP_K_AFTER_RERANK]],
            [x[2] for x in filtered[:TOP_K_AFTER_RERANK]]
        )
    
    reranker = get_reranker()
    
    # Score each chunk
    pairs = [[question, doc] for doc, _, _, _ in filtered]
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(
        zip(scores, filtered),
        key=lambda x: x[0],
        reverse=True
    )
    
    # Take top K
    top_k = ranked[:TOP_K_AFTER_RERANK]
    
    return (
        [x[1][0] for x in top_k],  # documents
        [x[1][1] for x in top_k],  # metadatas
        [x[1][2] for x in top_k]   # ids
    )
```

#### 4.5.3 Query and Answer Generation

```python
def query_rag(question: str) -> Dict[str, Any]:
    """
    Query the RAG system
    
    Process:
    1. Retrieve initial chunks (20)
    2. Apply two-stage filtering → 6 chunks
    3. Build context from filtered chunks
    4. Generate answer using appropriate QA method
    5. Return answer with metadata
    """
    
    # Get collection and embedder
    collection = get_collection()
    embedder = get_embedder()
    
    # Generate question embedding
    q_emb = embedder.encode([question])[0]
    
    # Retrieve initial chunks
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=INITIAL_RETRIEVAL_K,
        include=["documents", "metadatas", "distances"]
    )
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    ids = results['ids'][0]
    distances = results['distances'][0]
    
    # Apply two-stage filtering
    documents, metadatas, ids = apply_relevance_filtering(
        question, documents, metadatas, ids, distances
    )
    
    # Build context
    context = "\n\n".join(documents)
    
    # Generate answer based on QA_MODE
    if QA_MODE == "extractive":
        answer = generate_extractive_answer(question, context)
    elif QA_MODE == "generative":
        answer = generate_generative_answer(question, context)
    elif QA_MODE == "hybrid":
        answer = generate_hybrid_answer(question, context)
    
    return {
        'response': answer,
        'context': {'texts': documents},
        'retrieved_meta': metadatas
    }

def generate_generative_answer(question: str, context: str) -> str:
    """Generate answer using Llama"""
    gen_qa = get_generative_qa()
    
    # Clean context
    clean_context = context.replace("summary:", "").replace("table:", "").strip()
    
    # Llama-3.2 chat template
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant analyzing research papers. Provide thorough, detailed, 
and comprehensive answers to questions. Your answers should be substantial 
(around 100-120 words) and fully address all aspects of the question using 
information from the context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context from research paper:
{clean_context[:CONTEXT_MAX_CHARS]}

Question: {question}

Provide a detailed, comprehensive answer to the question using information 
from the context. Your answer should be thorough (around 100-120 words) and 
include all relevant information that addresses the question.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    # Generate
    result = gen_qa(
        prompt,
        max_new_tokens=ANSWER_MAX_LENGTH,
        min_new_tokens=ANSWER_MIN_LENGTH,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False
    )
    
    answer = result[0]["generated_text"].strip()
    return answer

def generate_hybrid_answer(question: str, context: str) -> str:
    """
    Hybrid: Try extractive first, fall back to generative
    
    Strategy:
    1. Try RoBERTa extractive QA
    2. Check if answer is high quality:
       - Confidence > 0.5
       - Length 15-100 words
    3. If good, return extractive answer
    4. Otherwise, use generative QA
    """
    
    # Try extractive
    extractive_qa = get_qa_pipeline()
    result = extractive_qa(
        question=question,
        context=context[:2000]  # RoBERTa limit
    )
    
    extractive_answer = result['answer']
    confidence = result['score']
    
    # Quality check
    words = extractive_answer.split()
    is_good = (
        confidence > 0.5 and
        len(words) >= 15 and
        len(words) <= 100
    )
    
    if is_good:
        return extractive_answer
    else:
        return generate_generative_answer(question, context)
```

### 4.6 Multi-PDF Pipeline (`multi_pdf_pipeline.py`)

**Purpose:** Extend RAG to handle multiple papers with comparison

**Key Differences from Single-PDF:**

1. **Separate ChromaDB collection** (`chroma_multi_store`)
2. **Source tracking** (each chunk tagged with paper_id)
3. **Per-paper querying** (filter by paper_id)

**Implementation:**

```python
def index_paper(pdf_path: str, paper_id: str, paper_title: str):
    """
    Index a single paper into multi-PDF collection
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Unique identifier (e.g., "paper1", "arxiv_2301_12345")
        paper_title: Human-readable title
    
    Process:
    1. Extract content from PDF
    2. Tag each chunk with paper_id and paper_title
    3. Store in shared multi-PDF collection
    """
    
    init_multi_chroma()
    
    # Extract content
    chunks = load_pdf_elements(pdf_path)
    
    embedder = get_embedder()
    
    for idx, chunk in enumerate(chunks):
        # Generate embedding
        embedding = embedder.encode(chunk['content'][:512])
        
        # Create unique ID
        chunk_id = f"{paper_id}_chunk_{idx}"
        
        # Add source metadata
        metadata = chunk['metadata']
        metadata['paper_id'] = paper_id
        metadata['paper_title'] = paper_title
        
        # Store in collection
        _multi_collection.add(
            ids=[chunk_id],
            embeddings=[embedding.tolist()],
            documents=[chunk['content']],
            metadatas=[metadata]
        )
        
        # Store in docstore
        _multi_docstore[chunk_id] = {
            'content': chunk['content'],
            'metadata': metadata,
            'paper_id': paper_id
        }

def query_single_paper(question: str, paper_id: str) -> Dict[str, Any]:
    """
    Query a specific paper in the multi-PDF collection
    
    Uses ChromaDB filtering to only retrieve chunks from target paper
    """
    
    embedder = get_embedder()
    q_emb = embedder.encode([question])[0]
    
    # Query with paper_id filter
    results = _multi_collection.query(
        query_embeddings=[q_emb],
        n_results=INITIAL_RETRIEVAL_K,
        where={"paper_id": paper_id},  # Only this paper!
        include=["documents", "metadatas", "distances"]
    )
    
    # Apply filtering and generate answer (same as single-PDF)
    # ... (similar to query_rag)
    
    return {
        'paper_id': paper_id,
        'paper_title': metadata['paper_title'],
        'answer': answer,
        'context': context_chunks
    }
```

### 4.7 Answer Comparator (`answer_comparator.py`)

**Purpose:** Compare answers from different papers using Llama as judge

**Implementation:**

```python
def llama_judge(
    question: str,
    answer1: Dict[str, Any],
    answer2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use Llama to judge which answer is better
    
    Criteria:
    1. Completeness (addresses all parts of question)
    2. Accuracy (correct information)
    3. Relevance (directly answers question)
    4. Clarity (well-explained)
    5. Concrete details (specific facts, not vague)
    
    Returns:
    {
        'winner': 'paper1' or 'paper2',
        'reasoning': 'explanation of decision',
        'criteria_scores': {
            'completeness': 'paper1' or 'paper2',
            ...
        }
    }
    """
    
    gen_qa = get_generative_qa()
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert judge comparing answers to research questions. 
Evaluate based on: completeness, accuracy, relevance, clarity, 
and concrete details.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

Answer from Paper 1 ("{answer1['paper_title']}"):
{answer1['answer']}

Answer from Paper 2 ("{answer2['paper_title']}"):
{answer2['answer']}

Compare these answers and determine which is better. Provide:
1. Winner (Paper 1 or Paper 2)
2. Detailed reasoning
3. Score each criterion

Format your response as:
WINNER: [Paper 1 or Paper 2]
REASONING: [detailed explanation]
COMPLETENESS: [1 or 2]
ACCURACY: [1 or 2]
RELEVANCE: [1 or 2]
CLARITY: [1 or 2]
DETAILS: [1 or 2]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    result = gen_qa(
        prompt,
        max_new_tokens=JUDGE_MAX_LENGTH,
        temperature=0.3,  # Lower temp for more consistent judging
        return_full_text=False
    )
    
    response = result[0]["generated_text"]
    
    # Parse response
    winner = "paper1" if "Paper 1" in response.split("WINNER:")[1].split("\n")[0] else "paper2"
    reasoning = response.split("REASONING:")[1].split("\n")[0].strip()
    
    return {
        'winner': winner,
        'reasoning': reasoning,
        'raw_response': response
    }

def compare_answers(
    question: str,
    answer1: Dict[str, Any],
    answer2: Dict[str, Any],
    method: str = "llama_judge"
) -> Dict[str, Any]:
    """
    Compare two answers using specified method
    
    Methods:
    - llama_judge: Use Llama to evaluate (most intelligent)
    - similarity: Compare embedding similarity to question
    - hybrid: Combine multiple methods
    """
    
    if method == "llama_judge":
        return llama_judge(question, answer1, answer2)
    
    elif method == "similarity":
        # Fallback: Embedding similarity
        embedder = get_embedder()
        q_emb = embedder.encode([question])[0]
        a1_emb = embedder.encode([answer1['answer']])[0]
        a2_emb = embedder.encode([answer2['answer']])[0]
        
        sim1 = cosine_similarity(q_emb, a1_emb)
        sim2 = cosine_similarity(q_emb, a2_emb)
        
        return {
            'winner': 'paper1' if sim1 > sim2 else 'paper2',
            'scores': {'paper1': sim1, 'paper2': sim2}
        }
```

---

## 5. THE RAG PIPELINE EXPLAINED

### 5.1 Indexing Phase (One-Time Setup)

```
Step 1: PDF Extraction
┌─────────────────┐
│ research.pdf    │
│ - 20 pages      │
│ - Text, tables  │
│ - 5 figures     │
└────────┬────────┘
         │ loader.py
         ▼
┌─────────────────────────────────────┐
│ Extracted Content                   │
│ - 45 text chunks                    │
│ - 3 table chunks                    │
│ - 5 image chunks                    │
│ Total: 53 chunks                    │
└────────┬────────────────────────────┘
         │
         ▼

Step 2: Preprocessing
┌─────────────────────────────────────┐
│ Text Chunks                         │
│ "The proposed architecture uses..." │
│ "Experiments were conducted on..."  │
└────────┬────────────────────────────┘
         │ summarizer.py (if long)
         ▼
┌─────────────────────────────────────┐
│ Processed Text                      │
│ summary: "Architecture uses conv..." │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Image Chunks                        │
│ [Base64 image data]                 │
└────────┬────────────────────────────┘
         │ vision.py
         ▼
┌─────────────────────────────────────┐
│ Image Captions                      │
│ "diagram showing CNN architecture"  │
└────────┬────────────────────────────┘
         │
         ▼

Step 3: Embedding Generation
┌─────────────────────────────────────┐
│ All Processed Chunks                │
│ - 53 chunks of text                 │
└────────┬────────────────────────────┘
         │ sentence-transformers
         ▼
┌─────────────────────────────────────┐
│ Vector Embeddings                   │
│ chunk_1: [0.2, -0.5, ..., 0.3]     │
│ chunk_2: [0.1, -0.3, ..., 0.4]     │
│ ... (53 embeddings of 384 dims)    │
└────────┬────────────────────────────┘
         │
         ▼

Step 4: Storage
┌─────────────────────────────────────┐
│ ChromaDB                            │
│ Collection: "multimodal_rag"        │
│ - IDs: chunk_0 ... chunk_52         │
│ - Embeddings: 53 x 384              │
│ - Documents: original text          │
│ - Metadata: page, type, etc.        │
└─────────────────────────────────────┘
```

**Time Complexity:**
- PDF extraction: ~2 seconds/page
- Image captioning: ~1 second/image
- Embedding generation: ~15ms/chunk
- Total for 20-page paper: ~1-2 minutes

### 5.2 Query Phase (Real-Time)

```
Step 1: Question Processing
┌─────────────────────────────────────┐
│ User Question                       │
│ "What datasets were used?"          │
└────────┬────────────────────────────┘
         │ embed question
         ▼
┌─────────────────────────────────────┐
│ Question Embedding                  │
│ [0.15, -0.42, ..., 0.28]           │
└────────┬────────────────────────────┘
         │
         ▼

Step 2: Initial Retrieval (Semantic Search)
┌─────────────────────────────────────┐
│ ChromaDB Search                     │
│ Find 20 most similar chunks         │
│ Metric: Cosine similarity           │
└────────┬────────────────────────────┘
         │ ~50ms
         ▼
┌─────────────────────────────────────┐
│ Top 20 Chunks (by similarity)       │
│ 1. "evaluated on ImageNet..." 0.85  │
│ 2. "dataset comprises 1M..." 0.82   │
│ ...                                 │
│ 20. "baseline model uses..." 0.45   │
└────────┬────────────────────────────┘
         │
         ▼

Step 3: Stage 1 Filtering (Similarity Threshold)
┌─────────────────────────────────────┐
│ Filter by threshold (0.2)           │
│ Keep chunks with similarity > 0.2   │
└────────┬────────────────────────────┘
         │ instant
         ▼
┌─────────────────────────────────────┐
│ Filtered Chunks                     │
│ 1. "evaluated on ImageNet..." 0.85  │
│ ...                                 │
│ 12. "training procedure..." 0.23    │
│ (12 chunks remain)                  │
└────────┬────────────────────────────┘
         │
         ▼

Step 4: Stage 2 Reranking (Cross-Encoder)
┌─────────────────────────────────────┐
│ Cross-Encoder Scoring               │
│ For each chunk:                     │
│   score = CrossEncoder(question, chunk) │
│ More accurate than embeddings       │
└────────┬────────────────────────────┘
         │ ~200ms (12 chunks)
         ▼
┌─────────────────────────────────────┐
│ Reranked Chunks                     │
│ 1. "dataset comprises 1M..." 9.2    │
│ 2. "evaluated on ImageNet..." 8.7   │
│ 3. "test set includes..." 7.3       │
│ 4. "following benchmarks..." 6.1    │
│ 5. "data preprocessing..." 5.8      │
│ 6. "validation split..." 4.2        │
│ (Top 6 selected)                    │
└────────┬────────────────────────────┘
         │
         ▼

Step 5: Context Building
┌─────────────────────────────────────┐
│ Combine Top 6 Chunks                │
│ "dataset comprises 1M images...     │
│  evaluated on ImageNet...           │
│  test set includes...               │
│  ..." (3000 chars total)            │
└────────┬────────────────────────────┘
         │
         ▼

Step 6: Answer Generation (Hybrid)
┌─────────────────────────────────────┐
│ Try Extractive QA (RoBERTa)         │
│ Input: question + context           │
│ Output: "ImageNet, COCO"            │
│ Confidence: 0.65                    │
│ Length: 3 words                     │
└────────┬────────────────────────────┘
         │ Quality check
         ▼
┌─────────────────────────────────────┐
│ Quality Check                       │
│ Confidence > 0.5? YES ✓             │
│ Length 15-100 words? NO ✗           │
│ Decision: Use generative QA         │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Generative QA (Llama)               │
│ Prompt: system + context + question │
│ Generate: max 250, min 100 tokens   │
└────────┬────────────────────────────┘
         │ ~2-3 seconds
         ▼
┌─────────────────────────────────────┐
│ Final Answer                        │
│ "The paper evaluated the model on   │
│  ImageNet (1.2M images, 1000        │
│  classes), COCO (330K images for    │
│  object detection), and a custom    │
│  dataset of 50K medical images.     │
│  ImageNet was used for              │
│  classification while COCO was      │
│  used for detection and             │
│  segmentation tasks."               │
│ (105 tokens)                        │
└─────────────────────────────────────┘
```

**Time Breakdown:**
- Embedding question: 15ms
- Retrieval from ChromaDB: 50ms
- Stage 1 filtering: <1ms
- Stage 2 reranking: 200ms
- Extractive QA attempt: 300ms
- Generative QA: 2-3s
- **Total: ~3 seconds**

---

## 6. MULTI-PDF COMPARISON

### 6.1 Use Case

**Scenario:** Comparing two papers on the same topic

```
Question: "What datasets were used for evaluation?"

Paper 1 (ImageNet Classification):
- Indexed as paper_id="paper1"
- 15 pages, 67 chunks

Paper 2 (Medical Imaging):
- Indexed as paper_id="paper2"
- 12 pages, 54 chunks

Both in same ChromaDB collection with source tags
```

### 6.2 Workflow

```
Step 1: Query Both Papers
┌─────────────────────────────────────┐
│ query_single_paper(q, "paper1")     │
│ → Answer 1: "ImageNet, COCO..."     │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ query_single_paper(q, "paper2")     │
│ → Answer 2: "ChestX-ray14, NIH..."  │
└─────────────────────────────────────┘

Step 2: Compare Answers
┌─────────────────────────────────────┐
│ llama_judge(question, ans1, ans2)   │
│                                     │
│ Llama evaluates:                    │
│ - Completeness                      │
│ - Accuracy                          │
│ - Relevance                         │
│ - Clarity                           │
│ - Concrete details                  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Comparison Result                   │
│ Winner: Paper 2                     │
│ Reasoning: "Paper 2 provides more   │
│ specific details about the medical  │
│ datasets, including patient counts, │
│ disease categories, and image       │
│ resolutions. Paper 1 mentions       │
│ datasets but lacks specifics."      │
└─────────────────────────────────────┘
```

### 6.3 Llama as Judge

**Why Llama instead of simple metrics?**

| Method | Pros | Cons |
|--------|------|------|
| **String similarity** | Fast, deterministic | Misses semantic meaning |
| **Embedding similarity** | Captures meaning | Can't explain reasoning |
| **Llama judge** | Nuanced evaluation, explains reasoning | Slower, requires LLM |

**Example judgment:**

```
Question: "How was the model evaluated?"

Answer 1: "accuracy and F1 score"
Answer 2: "The model was evaluated using accuracy (94.2%), 
           F1 score (0.91), and ROC-AUC (0.96) on the 
           test set of 10,000 images."

Llama Judge:
WINNER: Answer 2
REASONING: Answer 2 is significantly better because it provides 
concrete metrics (94.2% accuracy, 0.91 F1, 0.96 ROC-AUC) and 
specifies the test set size (10,000 images). Answer 1 only 
mentions metric names without values or context.
COMPLETENESS: 2
ACCURACY: 2
RELEVANCE: 2
CLARITY: 2
DETAILS: 2
```

---

## 7. QASPER EVALUATION SYSTEM

### 7.1 What is QASPER?

**QASPER:** Question Answering on Scientific Papers

- **Source:** Allen Institute for AI (AI2)
- **Papers:** 1,585 NLP research papers
- **Questions:** 5,049 questions with expert answers
- **Purpose:** Benchmark for scientific QA systems

**Why evaluate on QASPER?**
- Objective performance measurement
- Compare to published research
- Identify weaknesses
- Track improvements

### 7.2 QASPER Dataset Structure

```json
{
  "paper": {
    "title": "BERT: Pre-training of Deep Bidirectional...",
    "abstract": "We introduce a new language representation...",
    "full_text": {
      "Introduction": ["BERT stands for...", "Our model..."],
      "Method": ["We pre-train BERT using...", "..."],
      "Experiments": ["We evaluate on...", "..."]
    }
  },
  "qas": {
    "question": [
      "What pre-training tasks were used?",
      "What datasets were used for fine-tuning?"
    ],
    "answers": [
      {
        "answer": [
          {
            "free_form_answer": "masked language modeling and next sentence prediction",
            "evidence": ["We pre-train using two tasks: MLM and NSP"],
            "extractive_spans": ["masked language modeling", "next sentence prediction"]
          }
        ]
      }
    ]
  }
}
```

### 7.3 Evaluation Metrics

#### 7.3.1 F1 Score

**Definition:** Harmonic mean of precision and recall at word level

```python
def calculate_f1_score(prediction: str, reference: str) -> float:
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Precision = (words in both) / (words in prediction)
    Recall = (words in both) / (words in reference)
    """
    
    # Tokenize and normalize
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    
    # Find common tokens
    common = set(pred_tokens) & set(ref_tokens)
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

**Example:**

```
Prediction: "The model uses BERT and achieves 94% accuracy"
Reference:  "BERT-based model achieves 94.2% accuracy on test set"

Tokens (pred):  [model, uses, bert, achieves, 94, accuracy]
Tokens (ref):   [bert, based, model, achieves, 94.2, accuracy, test, set]
Common:         [model, bert, achieves, accuracy]

Precision: 4/6 = 0.667
Recall: 4/8 = 0.500
F1: 2 * (0.667 * 0.500) / (0.667 + 0.500) = 0.571
```

#### 7.3.2 ROUGE Scores

**Purpose:** Measure overlap between generated and reference text

**Types:**
- **ROUGE-1:** Unigram (single word) overlap
- **ROUGE-2:** Bigram (two consecutive words) overlap
- **ROUGE-L:** Longest common subsequence

**Example:**

```
Prediction: "The model was trained on ImageNet dataset"
Reference:  "Model trained on ImageNet"

ROUGE-1:
  Unigrams (pred): [the, model, was, trained, on, imagenet, dataset]
  Unigrams (ref):  [model, trained, on, imagenet]
  Overlap: [model, trained, on, imagenet]
  ROUGE-1: 4/4 = 1.0 (all reference words present)

ROUGE-2:
  Bigrams (pred): [the_model, model_was, was_trained, trained_on, on_imagenet, imagenet_dataset]
  Bigrams (ref):  [model_trained, trained_on, on_imagenet]
  Overlap: [trained_on, on_imagenet]
  ROUGE-2: 2/3 = 0.667

ROUGE-L:
  LCS: "model trained on imagenet" (4 words)
  ROUGE-L: 4/4 = 1.0
```

#### 7.3.3 BERTScore

**Purpose:** Semantic similarity using BERT embeddings

**How it works:**
1. Embed each token using BERT
2. Compute cosine similarity between embeddings
3. Match tokens greedily (max similarity)
4. Average similarities

**Advantage over F1/ROUGE:**
- Captures semantic similarity, not just exact matches
- "car" and "automobile" get high score
- F1 would give them 0 overlap

### 7.4 Evaluation Pipeline

```python
def evaluate_on_qasper(
    papers_data,
    num_questions: int = 50,
    apply_filtering: bool = True,
    use_hybrid_qa: bool = True
) -> Dict[str, Any]:
    """
    Run full evaluation on QASPER
    
    Process:
    1. Index all papers
    2. For each question:
       a. Query RAG system
       b. Compare to reference answer
       c. Calculate metrics
    3. Aggregate metrics
    4. Save results
    """
    
    # Index papers
    if not is_indexed():
        index_qasper_papers(papers_data)
    
    # Collect all questions
    all_questions = []
    for paper_idx, paper in enumerate(papers_data):
        for qa in paper['qas']:
            all_questions.append({
                'paper_id': str(paper_idx),
                'question': qa['question'],
                'reference_answer': qa['answers'][0]['free_form_answer'],
                'paper_title': paper['title']
            })
    
    # Sample questions if needed
    if num_questions:
        all_questions = all_questions[:num_questions]
    
    # Evaluate each question
    predictions = []
    all_metrics = []
    
    for qa in tqdm(all_questions):
        # Generate prediction
        pred_answer, context = query_qasper_rag(
            qa['question'],
            paper_id=qa['paper_id'],
            apply_filtering=apply_filtering,
            use_hybrid_qa=use_hybrid_qa
        )
        
        # Calculate metrics
        metrics = evaluate_answer(pred_answer, qa['reference_answer'])
        
        predictions.append({
            'question': qa['question'],
            'predicted': pred_answer,
            'reference': qa['reference_answer'],
            'metrics': metrics
        })
        
        all_metrics.append(metrics)
    
    # Aggregate
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in all_metrics[0].keys()
    }
    
    return {
        'num_questions': len(predictions),
        'average_metrics': avg_metrics,
        'predictions': predictions
    }
```

### 7.5 Typical Results

**Baseline (without optimizations):**
```
F1:           0.187
ROUGE-1:      0.210
ROUGE-2:      0.051
ROUGE-L:      0.134
BERTScore:    0.825
Pred Length:  ~110 tokens
Ref Length:   ~119 tokens
```

**After optimizations (target):**
```
F1:           0.22-0.26
ROUGE-1:      0.24-0.28
ROUGE-2:      0.06-0.08
ROUGE-L:      0.15-0.18
BERTScore:    0.840-0.860
Pred Length:  100-120 tokens
Ref Length:   ~119 tokens
```

---

## 8. PERFORMANCE OPTIMIZATION JOURNEY

### 8.1 Original System Issues

**Problem 1: Low Recall**
- Only retrieved 10 chunks initially
- Many relevant chunks missed
- Solution: Increase to 20 chunks

**Problem 2: Too Strict Filtering**
- Threshold 0.3 filtered out relevant chunks
- Context was too narrow
- Solution: Lower threshold to 0.2

**Problem 3: Limited Context**
- Only 1500 characters of context
- Not enough information for complex questions
- Solution: Increase to 3000 characters

**Problem 4: Paraphrasing Problem**
- Llama rephrases everything
- QASPER references are often direct quotes
- Low F1 due to different wording
- Solution: Hybrid extractive+generative

**Problem 5: Answer Length Mismatch**
- Generated answers too short (60-70 tokens)
- References average 119 tokens
- Missing information = low scores
- Solution: Increase min/max tokens, adjust prompt

### 8.2 Optimization Attempts

#### Attempt 1: Increase Retrieval

```python
# Before
INITIAL_RETRIEVAL_K = 10
SIMILARITY_THRESHOLD = 0.3
RERANKER_TOP_K = 4

# After
INITIAL_RETRIEVAL_K = 20  # +10 more chunks
SIMILARITY_THRESHOLD = 0.2  # More permissive
RERANKER_TOP_K = 6  # +2 more final chunks
```

**Expected:** Better recall, more relevant context
**Result:** Helped, but not enough

#### Attempt 2: Hybrid QA (Failed Initially)

```python
def hybrid_qa(question, context):
    # Try extractive first
    extractive_result = roberta_qa(question, context)
    
    # Accept if confidence > 0.2 and length > 4 words
    if extractive_result['score'] > 0.2 and len(extractive_result['answer'].split()) > 4:
        return extractive_result['answer']  # TOO PERMISSIVE!
    
    # Otherwise use generative
    return llama_qa(question, context)
```

**Expected:** Best of both worlds
**Result:** WORSE! Generated very short answers (60 tokens)
**Why:** Thresholds too low, accepted poor extractive answers

#### Attempt 3: Optimize for Conciseness (Failed)

```python
# Attempted "optimization"
ANSWER_MAX_LENGTH = 120  # Reduced from 200
ANSWER_MIN_LENGTH = 80   # Increased from 50

prompt = "Provide direct, concise answers using exact wording from context"
```

**Expected:** More focused answers with better overlap
**Result:** WORSE! Answers too short (65 tokens vs 119 needed)
**Why:** QASPER needs comprehensive answers, not concise ones

### 8.3 Current Best Configuration

```python
# Retrieval
INITIAL_RETRIEVAL_K = 20  # Good recall
SIMILARITY_THRESHOLD = 0.2  # Permissive enough
RERANKER_TOP_K = 6  # Adequate context

# Context
CONTEXT_MAX_CHARS = 3000  # Plenty of information

# Answer Generation
ANSWER_MAX_LENGTH = 250  # Allow complete answers
ANSWER_MIN_LENGTH = 100  # Ensure detail
temperature = 0.7  # Balance creativity and focus

# Prompt
"Provide thorough, detailed, comprehensive answers (100-120 words).
Include all relevant information from the context."

# Hybrid QA (if using)
extractive_confidence_threshold = 0.5  # Higher bar
extractive_min_length = 15  # Require substantial answers
```

### 8.4 Lessons Learned

**Lesson 1:** More context is better (up to model limits)
- 20 chunks >> 10 chunks
- 3000 chars >> 1500 chars

**Lesson 2:** Match reference answer characteristics
- QASPER refs: ~120 tokens → generate ~120 tokens
- QASPER refs: comprehensive → prompt for comprehensive

**Lesson 3:** Hybrid QA needs careful tuning
- Low thresholds accept bad extractive answers
- Need high confidence (>0.5) and good length (15+ words)

**Lesson 4:** Prompts matter enormously
- "Concise" → short answers → low F1
- "Comprehensive, detailed" → complete answers → better F1

**Lesson 5:** Evaluation drives optimization
- Without QASPER metrics, couldn't measure improvements
- Objective metrics reveal hidden problems

---

## 9. HOW TO USE (COMPLETE GUIDE)

### 9.1 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Document-AI.git
cd Document-AI

# Install dependencies
pip install -r requirements.txt

# For evaluation
pip install -r requirements_eval.txt

# System dependencies (Ubuntu/Colab)
apt-get update
apt-get install -y poppler-utils tesseract-ocr libmagic1
```

### 9.2 HuggingFace Authentication

**Why needed:** Llama models are gated (require acceptance of license)

```python
from huggingface_hub import login
import os

# Get token from https://huggingface.co/settings/tokens
HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxx"

# Login
login(token=HF_TOKEN)

# Set for subprocesses
os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGING_FACE_HUB_TOKEN'] = HF_TOKEN

# Accept Llama license at:
# https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

### 9.3 Single-PDF Usage

#### Index a PDF

```bash
python main.py --index --pdf papers/my_paper.pdf
```

**What happens:**
1. Extracts content from PDF
2. Processes text, tables, images
3. Generates embeddings
4. Stores in ChromaDB (./chroma_store/)
5. Creates docstore.json

**Time:** ~1-2 minutes for 20-page paper

#### Query the PDF

```bash
python main.py --query "What datasets were used for evaluation?"
```

**Output:**
```
================================================================================
STAGE 1: SIMILARITY-BASED FILTERING
================================================================================
Chunk 1: Similarity = 0.8245 [PASS]
Preview: The experiments were conducted on three datasets...
...

================================================================================
STAGE 2: CROSS-ENCODER RERANKING
================================================================================
Rank 1: Score = 8.3421 [SELECTED]
Preview: The experiments were conducted on three datasets...
...

Using device for generative QA: GPU
Loading Llama model: meta-llama/Llama-3.2-3B-Instruct

Answer:
The paper evaluated the model on three main datasets: ImageNet (1.2M training
images, 50K validation, 1000 classes), COCO (330K images for object detection
and segmentation), and a custom medical imaging dataset of 50K chest X-rays.
ImageNet was used for image classification tasks, COCO for object detection
and instance segmentation, and the medical dataset for disease classification.
The models achieved 94.2% top-1 accuracy on ImageNet, 45.3 mAP on COCO, and
92.1% accuracy on the medical dataset.

Retrieved metadata:
- Chunk from page 4, section: Experiments
- Chunk from page 5, section: Results
- Chunk from page 6, section: Datasets
================================================================================
```

### 9.4 Multi-PDF Usage

#### Reset Index

```bash
python main_multi.py --reset
```

#### Index Multiple Papers

```bash
# Paper 1
python main_multi.py --index papers/paper1.pdf paper1 "Deep Learning for Vision"

# Paper 2
python main_multi.py --index papers/paper2.pdf paper2 "Transformers in NLP"
```

#### Compare Answers

```bash
python main_multi.py --compare-query "What architecture was used?" \
    --paper1-id paper1 \
    --paper2-id paper2 \
    --method llama_judge
```

**Output:**
```
================================================================================
COMPARING ANSWERS FROM MULTIPLE PAPERS
================================================================================

Question: What architecture was used?

Paper 1 (Deep Learning for Vision):
The paper uses a ResNet-50 architecture with modifications including...

Paper 2 (Transformers in NLP):
The architecture is based on the Transformer model with 12 encoder layers...

================================================================================
LLAMA JUDGE COMPARISON
================================================================================

Winner: Paper 1

Reasoning:
Paper 1 provides more specific architectural details including the base
architecture (ResNet-50), specific modifications made, and justification
for design choices. Paper 2 mentions Transformers but provides less detail
about the specific configuration and modifications. Paper 1's answer is
more comprehensive and directly addresses the question with concrete
technical specifications.

Criteria Breakdown:
- Completeness: Paper 1
- Accuracy: Tie
- Relevance: Paper 1
- Clarity: Paper 1
- Concrete Details: Paper 1

================================================================================
```

### 9.5 QASPER Evaluation

#### Quick Evaluation (10 questions)

```bash
python evaluation/run_evaluation.py --num-questions 10 --filtering --save-results
```

#### Full Evaluation

```bash
python evaluation/run_evaluation.py --full --filtering --save-results
```

**Output files:**
```
evaluation_results/
├── qasper_predictions_with_filtering_hybrid.json
├── metrics_report.txt
└── comparison_chart.png
```

#### Compare Configurations

```bash
python evaluation/run_evaluation.py --compare --num-questions 50
```

**Output:**
```
================================================================================
CONFIGURATION COMPARISON
================================================================================

Metric                  | No Filtering | With Filtering | Improvement
--------------------------------------------------------------------------------
f1                      |      0.1823 |         0.2156 |     +18.27%
rouge1                  |      0.2034 |         0.2398 |     +17.89%
rouge2                  |      0.0489 |         0.0623 |     +27.40%
rougeL                  |      0.1289 |         0.1521 |     +18.00%
bert_score_f1           |      0.8234 |         0.8456 |      +2.70%
pred_length             |     65.3000 |       108.2000 |     +65.70%

================================================================================
```

### 9.6 Google Colab Usage

See the complete Colab script provided earlier. Key points:

**Cell Order:**
1. Mount Drive
2. Install dependencies (run once)
3. Authenticate HuggingFace (run once)
4. Setup project path
5. Load QASPER dataset
6. Import modules
7. Index papers (run once, ~15 min)
8. Run evaluation (~20 min for 50 questions)
9. Analyze results

**Tip:** After first run, you can skip indexing (Cell 7) since it's persistent

---

## 10. TROUBLESHOOTING

### 10.1 Common Issues

#### Issue 1: HuggingFace Authentication Error

**Error:**
```
GatedRepoError: 401 Client Error. Cannot access gated repo for url 
https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
```

**Solution:**
1. Get token: https://huggingface.co/settings/tokens
2. Accept license: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Set environment variables:
```python
os.environ['HF_TOKEN'] = "your_token"
os.environ['HUGGING_FACE_HUB_TOKEN'] = "your_token"
```
4. Restart kernel/runtime

#### Issue 2: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

**Option A: Use smaller batch size**
```python
# In config.py
EMBED_BATCH = 32  # Reduce from 64
```

**Option B: Use CPU for some models**
```python
# In rag_pipeline.py
def get_qa_pipeline():
    return pipeline("question-answering", model=QA_MODEL, device=-1)  # CPU
```

**Option C: Clear cache between operations**
```python
import torch
torch.cuda.empty_cache()
```

#### Issue 3: ChromaDB Collection Not Found

**Error:**
```
RuntimeError: Chroma collection not found. Run setup_retriever() first.
```

**Solution:**
```bash
# Re-index the PDF
python main.py --index --pdf your_paper.pdf

# Or for QASPER
# Re-run Cell 11 in Colab (index_qasper_papers)
```

#### Issue 4: Module Import Errors in Colab

**Error:**
```
ModuleNotFoundError: No module named 'evaluation.qasper_eval'
```

**Solution:**
```python
# Force reload modules
import sys
if 'evaluation.qasper_eval' in sys.modules:
    del sys.modules['evaluation.qasper_eval']

# Re-import
from evaluation.qasper_eval import evaluate_on_qasper
```

#### Issue 5: PDF Extraction Fails

**Error:**
```
ModuleNotFoundError: No module named 'pdfminer'
```

**Solution:**
```bash
# Install all PDF dependencies
pip install "unstructured[all-docs]" unstructured-inference pdfminer.six pi-heif pypdf pillow-heif

# System dependencies
apt-get install -y poppler-utils tesseract-ocr
```

#### Issue 6: Short Answers (Low F1)

**Problem:** Predictions are 60-70 tokens but references are 119 tokens

**Solution:**
```python
# In config.py
ANSWER_MAX_LENGTH = 250  # Increase from 120
ANSWER_MIN_LENGTH = 100  # Increase from 50

# In qasper_eval.py, check prompt encourages comprehensive answers
# Prompt should say: "Provide thorough, detailed answers (100-120 words)"
```

#### Issue 7: Slow Evaluation

**Problem:** Evaluation taking too long

**Optimization:**
1. Reduce num_questions for testing:
```bash
python evaluation/run_evaluation.py --num-questions 10  # Quick test
```

2. Disable reranking (faster but less accurate):
```python
# In config.py
ENABLE_RERANKING = False
```

3. Use smaller model for testing:
```python
# In config.py (temporarily)
GENERATIVE_QA_MODEL = "google/flan-t5-base"  # Faster than Llama
```

### 10.2 Performance Tuning

#### For Better Accuracy

```python
# config.py
INITIAL_RETRIEVAL_K = 30  # More chunks (better recall)
SIMILARITY_THRESHOLD = 0.15  # More permissive
RERANKER_TOP_K = 8  # More context
CONTEXT_MAX_CHARS = 4000  # More information
ANSWER_MAX_LENGTH = 300  # Allow longer answers
```

#### For Faster Speed

```python
# config.py
INITIAL_RETRIEVAL_K = 10  # Fewer chunks
ENABLE_RERANKING = False  # Skip reranking
RERANKER_TOP_K = 3  # Less context
ANSWER_MAX_LENGTH = 150  # Shorter answers
```

#### For Lower Memory Usage

```python
# Use smaller models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Already small
GENERATIVE_QA_MODEL = "google/flan-t5-base"  # 250M vs 3B params

# Reduce batch sizes
EMBED_BATCH = 16  # Reduce from 64

# Use CPU for some models
device = -1  # In pipeline() calls
```

---

## SUMMARY

This document has covered:

1. **Project Overview:** Intelligent research paper QA system
2. **Core Concepts:** RAG, embeddings, two-stage filtering
3. **Architecture:** Modular design with clear separation
4. **Implementation:** Detailed code explanations for each file
5. **RAG Pipeline:** Step-by-step indexing and querying
6. **Multi-PDF:** Comparison and Llama-as-judge
7. **QASPER Evaluation:** Scientific benchmarking
8. **Optimization:** Lessons learned and best practices
9. **Usage:** Complete guides for all modes
10. **Troubleshooting:** Common issues and solutions

**Key Takeaways:**

- RAG combines retrieval (search) with generation (LLMs)
- Two-stage filtering improves context quality significantly
- Hybrid QA can combine strengths of extractive and generative approaches
- Evaluation on benchmarks (QASPER) reveals hidden problems
- Configuration tuning is crucial for good performance
- Answer length and completeness matter for F1 scores

**Next Steps:**

1. Run evaluation on your papers
2. Analyze metrics to identify weaknesses
3. Tune configuration based on your specific use case
4. Consider fine-tuning models for your domain
5. Extend to multi-modal QA (leveraging images/tables more)

---

**Document Version:** 2.0  
**Last Updated:** January 2026  
**Maintained by:** Document-AI Team
