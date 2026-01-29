# multi_pdf_pipeline.py - Multi-PDF RAG pipeline with source tracking
import os
import uuid
import json
from typing import Dict, Any, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from config import (
    MULTI_CHROMA_PATH, EMBEDDING_MODEL_NAME, TOP_K, 
    ENABLE_RELEVANCE_FILTERING, SIMILARITY_THRESHOLD, ENABLE_RERANKING,
    RERANKER_MODEL, TOP_K_AFTER_RERANK, INITIAL_RETRIEVAL_K,
    QA_MODE, GENERATIVE_QA_MODEL, ANSWER_MAX_LENGTH, CONTEXT_MAX_CHARS
)
from loader import load_pdf_elements
from summarizer import summarize_text
from vision import summarize_image
from rag_pipeline import get_embedder, get_generative_qa, get_reranker

MULTI_DOCSTORE_FILE = "multi_docstore.json"

_multi_client: Optional[PersistentClient] = None
_multi_collection = None

def init_multi_chroma(persist_directory: str = MULTI_CHROMA_PATH):
    """Initialize multi-PDF ChromaDB collection."""
    global _multi_client, _multi_collection
    if _multi_client is None:
        _multi_client = PersistentClient(path=persist_directory)
    
    try:
        _multi_collection = _multi_client.get_collection("multi_pdf_rag")
    except Exception:
        _multi_collection = _multi_client.create_collection(name="multi_pdf_rag")
    return _multi_collection

def save_multi_docstore(docstore: Dict[str, Any], fname: str = MULTI_DOCSTORE_FILE):
    """Save multi-PDF docstore."""
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)

def load_multi_docstore(fname: str = MULTI_DOCSTORE_FILE) -> Dict[str, Any]:
    """Load multi-PDF docstore."""
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _reset_multi_collection(client: PersistentClient, collection_name: str = "multi_pdf_rag"):
    """Reset multi-PDF collection."""
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)

def index_paper(pdf_path: str, paper_id: str, paper_title: str) -> bool:
    """
    Index a single paper with source metadata.
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Unique identifier (e.g., "paper1", "paper2")
        paper_title: Human-readable title for display
    
    Returns:
        True if successful
    """
    global _multi_client, _multi_collection
    
    print(f"\nIndexing {paper_id}: {paper_title}")
    print(f"PDF: {pdf_path}")
    
    elements = load_pdf_elements(pdf_path)
    if not elements:
        raise RuntimeError(f"No elements extracted from {pdf_path}")
    
    # Load existing docstore
    docstore = load_multi_docstore()
    
    # Initialize/get collection
    if _multi_client is None:
        client = PersistentClient(path=MULTI_CHROMA_PATH)
        _multi_client = client
        try:
            _multi_collection = client.get_collection("multi_pdf_rag")
        except Exception:
            _multi_collection = client.create_collection(name="multi_pdf_rag")
    else:
        client = _multi_client
    
    collection = _multi_collection
    embedder = get_embedder()
    
    texts_to_embed: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []
    
    # Process elements with source metadata
    for el in elements:
        doc_id = str(uuid.uuid4())
        el_type = el.get("type", "text")
        
        # Add source metadata to all chunks
        base_metadata = {
            "doc_id": doc_id,
            "source": paper_id,
            "paper_title": paper_title,
            "type": el_type,
            "page": el.get("meta", {}).get("page_number")
        }
        
        if el_type in ("text", "table"):
            summary = summarize_text(el["content"])
            texts_to_embed.append(summary)
            metadatas.append(base_metadata)
            ids.append(doc_id)
            docstore[doc_id] = {
                "type": el_type,
                "original": el["content"],
                "source": paper_id,
                "paper_title": paper_title
            }
        elif el_type == "image":
            summary = summarize_image(el["content"], surrounding_text=None)
            texts_to_embed.append(summary)
            metadatas.append(base_metadata)
            ids.append(doc_id)
            docstore[doc_id] = {
                "type": "image",
                "original": el["content"],
                "source": paper_id,
                "paper_title": paper_title
            }
        else:
            summary = summarize_text(str(el.get("content", "")))
            texts_to_embed.append(summary)
            metadatas.append(base_metadata)
            ids.append(doc_id)
            docstore[doc_id] = {
                "type": "text",
                "original": el.get("content", ""),
                "source": paper_id,
                "paper_title": paper_title
            }
    
    # Compute embeddings
    embeddings = []
    batch_size = 32
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.extend(embs)
    
    # Add to collection
    collection.add(
        documents=texts_to_embed,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    
    # Save docstore
    save_multi_docstore(docstore)
    
    print(f"✓ Indexed {len(texts_to_embed)} chunks from {paper_id}")
    return True

def apply_relevance_filtering_multi(
    question: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    distances: List[float],
    paper_id: str
) -> tuple:
    """Two-stage filtering for multi-PDF mode."""
    if not ENABLE_RELEVANCE_FILTERING:
        return documents[:TOP_K], metadatas[:TOP_K], ids[:TOP_K]
    
    print(f"\n{'='*80}")
    print(f"STAGE 1: SIMILARITY FILTERING - {paper_id.upper()}")
    print(f"{'='*80}")
    
    # Stage 1: Similarity threshold
    filtered_docs = []
    filtered_metas = []
    filtered_ids = []
    filtered_scores = []
    
    for idx, (doc, meta, doc_id, distance) in enumerate(zip(documents, metadatas, ids, distances)):
        similarity_score = 1.0 / (1.0 + distance)
        
        status = "PASS" if similarity_score >= SIMILARITY_THRESHOLD else "FILTERED"
        print(f"Chunk {idx+1}: Sim={similarity_score:.4f} [{status}] {doc[:100]}...")
        
        if similarity_score >= SIMILARITY_THRESHOLD:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_ids.append(doc_id)
            filtered_scores.append(similarity_score)
    
    print(f"\nStage 1: {len(documents)} → {len(filtered_docs)} chunks")
    
    if not filtered_docs:
        print("Warning: No chunks passed filtering, using top results")
        return documents[:TOP_K], metadatas[:TOP_K], ids[:TOP_K]
    
    # Stage 2: Cross-encoder reranking
    if not ENABLE_RERANKING or len(filtered_docs) <= TOP_K_AFTER_RERANK:
        return filtered_docs[:TOP_K_AFTER_RERANK], filtered_metas[:TOP_K_AFTER_RERANK], filtered_ids[:TOP_K_AFTER_RERANK]
    
    print(f"\n{'='*80}")
    print(f"STAGE 2: CROSS-ENCODER RERANKING - {paper_id.upper()}")
    print(f"{'='*80}")
    
    reranker = get_reranker()
    pairs = [[question, doc] for doc in filtered_docs]
    rerank_scores = reranker.predict(pairs)
    
    scored_results = list(zip(rerank_scores, filtered_docs, filtered_metas, filtered_ids))
    scored_results.sort(reverse=True, key=lambda x: x[0])
    
    for idx, (score, doc, meta, doc_id) in enumerate(scored_results):
        selected = "SELECTED" if idx < TOP_K_AFTER_RERANK else ""
        print(f"Rank {idx+1}: Score={score:.4f} [{selected}] {doc[:100]}...")
    
    reranked_docs = [item[1] for item in scored_results[:TOP_K_AFTER_RERANK]]
    reranked_metas = [item[2] for item in scored_results[:TOP_K_AFTER_RERANK]]
    reranked_ids = [item[3] for item in scored_results[:TOP_K_AFTER_RERANK]]
    
    print(f"\nStage 2: {len(filtered_docs)} → {len(reranked_docs)} chunks selected\n")
    
    return reranked_docs, reranked_metas, reranked_ids

def query_single_paper(question: str, paper_id: str) -> Dict[str, Any]:
    """
    Query a single paper with full filtering pipeline.
    
    Returns:
        {
            "response": str,
            "context": {"texts": [...], "images": [...]},
            "retrieved_meta": [...],
            "paper_id": str,
            "paper_title": str
        }
    """
    client = _multi_client or PersistentClient(path=MULTI_CHROMA_PATH)
    try:
        collection = client.get_collection("multi_pdf_rag")
    except Exception:
        raise RuntimeError("Multi-PDF collection not found. Run indexing first.")
    
    embedder = get_embedder()
    docstore = load_multi_docstore()
    
    retrieval_k = INITIAL_RETRIEVAL_K if ENABLE_RELEVANCE_FILTERING else TOP_K
    
    # Query with source filter
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=retrieval_k,
        where={"source": paper_id},  # Filter by paper
        include=["documents", "metadatas", "distances"]
    )
    
    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(documents)
    
    if not documents:
        return {
            "response": f"No content found in {paper_id}",
            "context": {"texts": [], "images": []},
            "retrieved_meta": [],
            "paper_id": paper_id,
            "paper_title": metadatas[0].get("paper_title", paper_id) if metadatas else paper_id
        }
    
    paper_title = metadatas[0].get("paper_title", paper_id)
    
    # Apply filtering
    if ENABLE_RELEVANCE_FILTERING:
        documents, metadatas, ids = apply_relevance_filtering_multi(
            question, documents, metadatas, ids, distances, paper_id
        )
    
    # Build context
    contexts: List[str] = []
    images_b64: List[str] = []
    retrieved_meta: List[Dict[str, Any]] = []
    
    for doc_text, md, idx in zip(documents, metadatas, ids):
        retrieved_meta.append(md)
        doc_id = md.get("doc_id", idx)
        entry = docstore.get(doc_id, {})
        if entry.get("type") == "image":
            contexts.append(doc_text)
            images_b64.append(entry.get("original"))
        else:
            contexts.append(doc_text)
    
    context_text = "\n\n".join(contexts).strip()
    
    if not context_text:
        answer = "No relevant context found."
    else:
        # Generate answer with Llama
        gen_qa = get_generative_qa()
        clean_context = context_text.replace("summary:", "").replace("summarize:", "").strip()
        
        if "llama" in GENERATIVE_QA_MODEL.lower():
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant analyzing the research paper titled "{paper_title}". Answer questions based on the provided context from this paper. Synthesize information and provide clear, comprehensive answers.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context from {paper_title}:
{clean_context[:CONTEXT_MAX_CHARS]}

Question: {question}

Provide a clear and comprehensive answer based on the context above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            result = gen_qa(
                prompt,
                max_new_tokens=ANSWER_MAX_LENGTH,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                return_full_text=False
            )
            answer = result[0]["generated_text"].strip()
        else:
            # T5/FLAN fallback
            prompt = f"question: {question} context: {clean_context[:1000]}"
            result = gen_qa(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_p=0.95
            )
            answer = result[0]["generated_text"].strip()
    
    return {
        "response": answer,
        "context": {"texts": contexts, "images": images_b64},
        "retrieved_meta": retrieved_meta,
        "paper_id": paper_id,
        "paper_title": paper_title
    }

def reset_multi_index():
    """Reset the multi-PDF index (clear all papers)."""
    client = PersistentClient(path=MULTI_CHROMA_PATH)
    _reset_multi_collection(client)
    
    # Clear docstore
    if os.path.exists(MULTI_DOCSTORE_FILE):
        os.remove(MULTI_DOCSTORE_FILE)
    
    global _multi_client, _multi_collection
    _multi_client = client
    try:
        _multi_collection = client.get_collection("multi_pdf_rag")
    except Exception:
        _multi_collection = client.create_collection(name="multi_pdf_rag")
    
    print("Multi-PDF index reset successfully")
