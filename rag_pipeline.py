# rag_pipeline.py RAG llogic index + query
import os
import uuid
import json
from typing import Dict, Any, List, Optional

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import torch
from transformers import pipeline
from config import CHROMA_PATH, EMBEDDING_MODEL_NAME, TOP_K, QA_MODEL, TEXT_SUMMARIZER
from loader import load_pdf_elements
from summarizer import summarize_text
from vision import summarize_image

DOCSTORE_FILE = "docstore.json"

# lazy inits
_embedder: Optional[SentenceTransformer] = None # Optional is type hinting for variable that can be none or of given type
_qa_pipeline = None
_client: Optional[PersistentClient] = None
_collection = None

def init_chroma(persist_directory: str = CHROMA_PATH):
    """
    Initialize or return a PersistentClient and ensure collection exists.
    """
    global _client, _collection
    if _client is None: # Loading the chroma client only once
        _client = PersistentClient(path=persist_directory)

    # Ensure collection exists (delete/create is handled by setup_retriever as needed)
    try:
        _collection = _client.get_collection("multimodal_rag")
    except Exception:
        _collection = _client.create_collection(name="multimodal_rag")
    return _collection

def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading embedding model on device: {device}")

    _embedder = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=device
    )
    return _embedder

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        # Extractive QA pipeline on GPU (device=0)
        _qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=0)
    return _qa_pipeline

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
        # Ignore if not exists or deletion fails
        pass
    # Create and return new collection
    return client.create_collection(name=collection_name)

def setup_retriever(pdf_path: Optional[str] = None, use_alternate_loader: bool = False) -> bool:
    """
    Extract -> summarize -> embed -> store in Chroma.
    Returns True if indexing succeeded.
    """
    elements = load_pdf_elements(pdf_path, use_alternate_loader=use_alternate_loader)
    if not elements:
        raise RuntimeError("No elements extracted from PDF. Check file path and extraction.")

    client = PersistentClient(path=CHROMA_PATH)
    # Reset/create collection
    collection = _reset_collection(client, collection_name="multimodal_rag")

    embedder = get_embedder()
    docstore: Dict[str, Any] = {}

    texts_to_embed: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    # Process elements
    for el in elements:
        doc_id = str(uuid.uuid4())
        el_type = el.get("type", "text")
        if el_type in ("text", "table"):
            # Summarizer (FLAN-T5) is used in summarizer.summarize_text
            summary = summarize_text(el["content"])
            texts_to_embed.append(summary)
            metadatas.append({"doc_id": doc_id, "type": el_type, "page": el.get("meta", {}).get("page_number")})
            ids.append(doc_id)
            docstore[doc_id] = {"type": el_type, "original": el["content"]}
        elif el_type == "image":
            # Get BLIP caption (vision.summarize_image)
            # Optionally pass surrounding text later for better captions
            summary = summarize_image(el["content"], surrounding_text=None)
            texts_to_embed.append(summary)
            metadatas.append({"doc_id": doc_id, "type": "image"})
            ids.append(doc_id)
            docstore[doc_id] = {"type": "image", "original": el["content"]}
        else:
            # Fallback treat as text
            summary = summarize_text(str(el.get("content", "")))
            texts_to_embed.append(summary)
            metadatas.append({"doc_id": doc_id, "type": "text"})
            ids.append(doc_id)
            docstore[doc_id] = {"type": "text", "original": el.get("content", "")}

    # Compute embeddings in batches
    embeddings = []
    batch_size = 32
    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i+batch_size]
        embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        embeddings.extend(embs)

    # Add to chroma collection
    collection.add(
        documents=texts_to_embed,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )

    # Save docstore
    save_docstore(docstore)
    # Keep client/collection globals for later query usage
    global _client, _collection
    _client = client
    _collection = collection

    return True

def query_rag(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Retrieve top_k results and run extractive QA on concatenated context.
    Returns {response, context:{texts, images}, retrieved_meta}
    """
    client = _client or PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection("multimodal_rag")
    except Exception:
        # No index yet
        raise RuntimeError("Chroma collection not found. Run setup_retriever() / --index first.")

    embedder = get_embedder()
    qa = get_qa_pipeline()
    docstore = load_docstore()

    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas"]
        )
    
    documents = results.get("documents", [[]])[0] if results.get("documents") else []
    metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    ids = results.get("ids", [[]])[0] if results.get("ids") else []

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
        qa_input = {"question": question, "context": context_text}
        res = qa(qa_input)  # Extractive QA
        answer = res.get("answer", "").strip()
        if not answer:
            # Fallback to returning the top retrieved snippet
            answer = contexts[0] if contexts else "No answer found."

    return {
        "response": answer,
        "context": {"texts": contexts, "images": images_b64},
        "retrieved_meta": retrieved_meta
    }
