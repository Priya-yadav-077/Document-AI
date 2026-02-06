# rag_pipeline.py RAG llogic index + query
import os
import uuid
import json
from typing import Dict, Any, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb import PersistentClient
from transformers import pipeline
from config import (
    CHROMA_PATH, EMBEDDING_MODEL_NAME, TOP_K, QA_MODEL, TEXT_SUMMARIZER,
    ENABLE_RELEVANCE_FILTERING, SIMILARITY_THRESHOLD, ENABLE_RERANKING,
    RERANKER_MODEL, TOP_K_AFTER_RERANK, INITIAL_RETRIEVAL_K,
    QA_MODE, GENERATIVE_QA_MODEL, ANSWER_MAX_LENGTH, ANSWER_MIN_LENGTH,
    CONTEXT_MAX_CHARS
)
from loader import load_pdf_elements
from summarizer import summarize_text
from vision import summarize_image

DOCSTORE_FILE = "docstore.json"

# lazy inits
_embedder: Optional[SentenceTransformer] = None
_qa_pipeline = None
_generative_qa = None
_reranker = None
_client: Optional[PersistentClient] = None
_collection = None

def init_chroma(persist_directory: str = CHROMA_PATH):
    """
    Initialize or return a PersistentClient and ensure collection exists.
    """
    global _client, _collection
    if _client is None: #loading the chroma client only once
        _client = PersistentClient(path=persist_directory)

    # ensure collection exists (delete/create is handled by setup_retriever as needed)
    try:
        _collection = _client.get_collection("multimodal_rag")
    except Exception:
        _collection = _client.create_collection(name="multimodal_rag")
    return _collection

def get_embedder():
    global _embedder
    if _embedder is None:
        # Auto-detect device (GPU if available, else CPU)
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device for embeddings: {device}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _embedder

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device for QA: {'GPU' if device == 0 else 'CPU'}")
        _qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=device)
    return _qa_pipeline

def get_generative_qa():
    global _generative_qa
    if _generative_qa is None:
        import torch
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device for generative QA: {'GPU' if device == 0 else 'CPU'}")
        
        # Check if model is Llama (needs text-generation pipeline)
        if "llama" in GENERATIVE_QA_MODEL.lower():
            print(f"Loading Llama model: {GENERATIVE_QA_MODEL}")
            _generative_qa = pipeline(
                "text-generation",
                model=GENERATIVE_QA_MODEL,
                device=device,
                model_kwargs={
                    "torch_dtype": torch.float16 if device >= 0 else torch.float32,
                    "low_cpu_mem_usage": True
                }
            )
        else:
            # T5/FLAN models use text2text-generation
            _generative_qa = pipeline(
                "text2text-generation",
                model=GENERATIVE_QA_MODEL,
                device=device
            )
    return _generative_qa

def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device for reranker: {device}")
        _reranker = CrossEncoder(RERANKER_MODEL, device=device)
    return _reranker

def save_docstore(docstore: Dict[str, Any], fname: str = DOCSTORE_FILE):
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(docstore, f, ensure_ascii=False, indent=2)

def load_docstore(fname: str = DOCSTORE_FILE) -> Dict[str, Any]:
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _reset_collection(client: PersistentClient, collection_name: str = "multimodal_rag"):
    """
    Delete the collection if exists and recreate it fresh.
    """
    try:
        client.delete_collection(collection_name)
    except Exception:
        # ignore if not exists or deletion fails
        pass
    # create and return new collection
    return client.create_collection(name=collection_name)

def setup_retriever(pdf_path: Optional[str] = None, use_alternate_loader: bool = False) -> bool:
    """
    Extract -> summarize -> embed -> store in Chroma.
    pdf_path: explicit path to PDF file, if None uses config default
    Returns True if indexing succeeded.
    """
    elements = load_pdf_elements(pdf_path, use_alternate_loader=use_alternate_loader)
    if not elements:
        raise RuntimeError("No elements extracted from PDF. Check file path and extraction.")

    client = PersistentClient(path=CHROMA_PATH)
    # reset/create collection
    collection = _reset_collection(client, collection_name="multimodal_rag")

    embedder = get_embedder()
    docstore: Dict[str, Any] = {}

    texts_to_embed: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    # process elements
    for el in elements:
        doc_id = str(uuid.uuid4())
        el_type = el.get("type", "text")
        if el_type in ("text", "table"):
            # summarizer (FLAN-T5) is used in summarizer.summarize_text
            summary = summarize_text(el["content"])
            texts_to_embed.append(summary)
            metadatas.append({"doc_id": doc_id, "type": el_type, "page": el.get("meta", {}).get("page_number")})
            ids.append(doc_id)
            docstore[doc_id] = {"type": el_type, "original": el["content"]}
        elif el_type == "image":
            # get BLIP caption (vision.summarize_image)
            # optionally pass surrounding text later for better captions
            summary = summarize_image(el["content"], surrounding_text=None)
            texts_to_embed.append(summary)
            metadatas.append({"doc_id": doc_id, "type": "image"})
            ids.append(doc_id)
            docstore[doc_id] = {"type": "image", "original": el["content"]}
        else:
            # fallback treat as text
            summary = summarize_text(str(el.get("content", "")))
            texts_to_embed.append(summary)
            metadatas.append({"doc_id": doc_id, "type": "text"})
            ids.append(doc_id)
            docstore[doc_id] = {"type": "text", "original": el.get("content", "")}

    # compute embeddings in batches
    embeddings = []
    batch_size = 32
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.extend(embs)

    # add to chroma collection
    collection.add(
        documents=texts_to_embed,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    # save docstore
    save_docstore(docstore)
    # keep client/collection globals for later query usage
    global _client, _collection
    _client = client
    _collection = collection

    return True

def apply_relevance_filtering(
    question: str,
    documents: List[str],
    metadatas: List[Dict[str, Any]],
    ids: List[str],
    distances: List[float]
) -> tuple:
    """
    Two-stage relevance filtering:
    Stage 1: Filter by cosine similarity threshold
    Stage 2: Rerank with cross-encoder and select top-K
    """
    if not ENABLE_RELEVANCE_FILTERING:
        return documents[:TOP_K], metadatas[:TOP_K], ids[:TOP_K]
    
    print("\n" + "="*80)
    print("STAGE 1: SIMILARITY-BASED FILTERING")
    print("="*80)
    
    # Stage 1: Score-based filtering
    filtered_docs = []
    filtered_metas = []
    filtered_ids = []
    filtered_scores = []
    
    for idx, (doc, meta, doc_id, distance) in enumerate(zip(documents, metadatas, ids, distances)):
        similarity_score = 1.0 / (1.0 + distance)
        
        status = "PASS" if similarity_score >= SIMILARITY_THRESHOLD else "FILTERED OUT"
        print(f"\nChunk {idx+1}: Similarity = {similarity_score:.4f} [{status}]")
        print(f"Preview: {doc[:150]}...")
        
        if similarity_score >= SIMILARITY_THRESHOLD:
            filtered_docs.append(doc)
            filtered_metas.append(meta)
            filtered_ids.append(doc_id)
            filtered_scores.append(similarity_score)
    
    print(f"\n{'-'*80}")
    print(f"Stage 1 Result: {len(documents)} -> {len(filtered_docs)} chunks passed (threshold: {SIMILARITY_THRESHOLD})")
    print(f"{'-'*80}\n")
    
    if not filtered_docs:
        print("Warning: No chunks passed Stage 1 filtering. Returning top results anyway.")
        return documents[:TOP_K], metadatas[:TOP_K], ids[:TOP_K]
    
    # Stage 2: Cross-encoder reranking
    if not ENABLE_RERANKING or len(filtered_docs) <= TOP_K_AFTER_RERANK:
        return filtered_docs[:TOP_K_AFTER_RERANK], filtered_metas[:TOP_K_AFTER_RERANK], filtered_ids[:TOP_K_AFTER_RERANK]
    
    print("="*80)
    print("STAGE 2: CROSS-ENCODER RERANKING")
    print("="*80)
    
    reranker = get_reranker()
    
    pairs = [[question, doc] for doc in filtered_docs]
    rerank_scores = reranker.predict(pairs)
    
    scored_results = list(zip(rerank_scores, filtered_docs, filtered_metas, filtered_ids))
    scored_results.sort(reverse=True, key=lambda x: x[0])
    
    print(f"\nReranking scores for {len(scored_results)} chunks:\n")
    for idx, (score, doc, meta, doc_id) in enumerate(scored_results):
        selected = "SELECTED" if idx < TOP_K_AFTER_RERANK else "NOT SELECTED"
        print(f"Rank {idx+1}: Score = {score:.4f} [{selected}]")
        print(f"Preview: {doc[:150]}...")
        print()
    mean_reranked_scores = np.mean([item[0] for item in scored_results[:TOP_K_AFTER_RERANK]])
    # Get avg margin to next m candidate scores
    marg_scores = []
    for cand_idx, (score, doc, meta, doc_id) in enumerate(scored_results[TOP_K_AFTER_RERANK:]):
        marg_scores.append(mean_reranked_scores - score)
    avg_margin = np.mean(marg_scores) if marg_scores else 0.0
    # print(f"\nAverage reranker score for selected top-{TOP_K_AFTER_RERANK}: {mean_reranked_scores:.4f}")
    print(f"Average margin to next {cand_idx} candidates: {avg_margin:.4f}")

    reranked_docs = [item[1] for item in scored_results[:TOP_K_AFTER_RERANK]]
    reranked_metas = [item[2] for item in scored_results[:TOP_K_AFTER_RERANK]]
    reranked_ids = [item[3] for item in scored_results[:TOP_K_AFTER_RERANK]]
    
    print(f"{'-'*80}")
    print(f"Stage 2 Result: {len(filtered_docs)} -> {len(reranked_docs)} chunks selected (top-K: {TOP_K_AFTER_RERANK})")
    print(f"{'-'*80}\n")
    
    return reranked_docs, reranked_metas, reranked_ids

def query_rag(question: str, top_k: int = None) -> Dict[str, Any]:
    """
    Retrieve documents, apply two-stage filtering, and generate answer.
    Returns {response, context:{texts, images}, retrieved_meta}
    """
    client = _client or PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection("multimodal_rag")
    except Exception:
        raise RuntimeError("Chroma collection not found. Run setup_retriever() / --index first.")

    embedder = get_embedder()
    docstore = load_docstore()
    
    # Use configured values if not specified
    retrieval_k = INITIAL_RETRIEVAL_K if ENABLE_RELEVANCE_FILTERING else (top_k or TOP_K)

    # Retrieve initial set of documents with distances
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=retrieval_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(documents)

    # Apply two-stage relevance filtering
    if ENABLE_RELEVANCE_FILTERING and documents:
        documents, metadatas, ids = apply_relevance_filtering(
            question, documents, metadatas, ids, distances
        )

    # Build context from filtered documents
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
        if QA_MODE == "extractive":
            # Pure extractive QA with RoBERTa
            qa = get_qa_pipeline()
            qa_input = {"question": question, "context": context_text}
            res = qa(qa_input)
            answer = res.get("answer", "").strip()
            if not answer:
                answer = contexts[0][:200] if contexts else "No answer found."
        
        elif QA_MODE == "generative":
            # Generative QA with LLM (Llama or T5)
            gen_qa = get_generative_qa()
            
            # Clean context - remove "summary:" and "summarize:" prefixes
            clean_context = context_text.replace("summary:", "").replace("summarize:", "").strip()
            
            # Check if using Llama (needs different prompting)
            if "llama" in GENERATIVE_QA_MODEL.lower():
                # Llama chat format with system + user messages
                prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are a helpful AI assistant analyzing research papers. Answer questions based on the provided context. Synthesize information and provide clear, comprehensive answers.<|eot_id|><|start_header_id|>user<|end_header_id|>

                Context:
                {clean_context[:CONTEXT_MAX_CHARS]}

                Question: {question}

                Provide a clear and comprehensive answer based on the context above. In case of a Yes/No answer, please restrict the answer to a single word: "Yes" or "No".<|eot_id|><|start_header_id|>assistant<|end_header_id|>

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
                # T5/FLAN models use simpler format
                prompt = f"question: {question} context: {clean_context[:1000]}"
                result = gen_qa(
                    prompt,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    repetition_penalty=2.0
                )
                answer = result[0]["generated_text"].strip()
        
        elif QA_MODE == "hybrid":
            # Hybrid: Extract with RoBERTa, then add relevant context
            qa = get_qa_pipeline()
            qa_input = {"question": question, "context": context_text}
            res = qa(qa_input)
            extracted_answer = res.get("answer", "").strip()
            
            if not extracted_answer or len(extracted_answer.split()) < 3:
                # If extraction failed or too short, use first context chunk
                answer = contexts[0][:300] if contexts else "No answer found."
            else:
                # Find which context chunk contains the extracted answer
                source_context = None
                for ctx in contexts:
                    if extracted_answer.lower() in ctx.lower():
                        source_context = ctx
                        break
                
                if not source_context:
                    source_context = contexts[0]
                
                # Build a readable answer with extracted fact + context
                # Remove redundant parts and create clean explanation
                context_snippet = source_context[:400].strip()
                
                # Clean answer: Main answer + relevant context
                if extracted_answer in context_snippet:
                    # Answer is already in context, use context directly
                    answer = context_snippet
                else:
                    # Combine answer with context
                    answer = f"{extracted_answer}. {context_snippet}"
                
                # Ensure answer is at least somewhat informative
                if len(answer.split()) < 15:
                    # Add more context if answer is too short
                    if len(contexts) > 1:
                        answer = f"{answer} Additionally, {contexts[1][:200]}"
        
        else:
            # Default to extractive if invalid mode
            qa = get_qa_pipeline()
            qa_input = {"question": question, "context": context_text}
            res = qa(qa_input)
            answer = res.get("answer", "").strip()
            if not answer:
                answer = contexts[0][:200] if contexts else "No answer found."

    return {
        "response": answer,
        "context": {"texts": contexts, "images": images_b64},
        "retrieved_meta": retrieved_meta
    }
