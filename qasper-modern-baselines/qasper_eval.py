"""
QASPER Dataset Evaluation Module

This module handles loading, converting, and evaluating on the QASPER dataset.
It reuses the existing RAG pipeline without modifying it.

QASPER Dataset: https://huggingface.co/datasets/allenai/qasper
- 1,585 NLP papers
- 5,049 questions with expert-written answers
- Abstractive and extractive questions
"""

import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# Add parent directory to path to import from main project
sys.path.insert(0, str(Path(__file__).parent.parent))


from config import (
    EMBEDDING_MODEL_NAME,
    SIMILARITY_THRESHOLD,
    RERANKER_TOP_K,
    CONTEXT_MAX_CHARS,
    ANSWER_MAX_LENGTH,
    ANSWER_MIN_LENGTH,
    RERANKER_MODEL,
    INITIAL_RETRIEVAL_K
)
from rag_pipeline import get_embedder, get_generative_qa, get_reranker, get_qa_pipeline
from summarizer import get_summarizer
from evaluation.evaluation_metrics import evaluate_answer, evaluate_batch

# Global variables for QASPER-specific ChromaDB
_qasper_client = None
_qasper_collection = None
_qasper_docstore = {}
_qasper_embedding_model = None


def load_qasper_dataset(split: str = "validation", num_papers: Optional[int] = None):
    """
    Load QASPER dataset from HuggingFace.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        num_papers: Number of papers to load (None = all)
    
    Returns:
        List of paper dictionaries
    """
    try:
        from datasets import load_dataset
        
        print(f"Loading QASPER dataset (split={split})...")
        
        # Load dataset (works with datasets<3.0.0)
        dataset = load_dataset("allenai/qasper", split=split)
        
        if num_papers:
            dataset = dataset.select(range(min(num_papers, len(dataset))))
        
        print(f"Loaded {len(dataset)} papers from QASPER")
        return dataset
    
    except Exception as e:
        print(f"Error loading QASPER dataset: {e}")
        print("\nThe datasets library version is incompatible.")
        print("Please run: !pip install datasets==2.18.0")
        raise


def convert_qasper_paper_to_chunks(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert QASPER paper format to text chunks compatible with RAG pipeline.
    
    QASPER paper structure:
    {
        'title': str,
        'abstract': str,
        'full_text': {
            'section_name': [paragraph1, paragraph2, ...],
            ...
        }
    }
    
    Output format (same as loader.py produces):
    [
        {'type': 'text', 'content': '...', 'metadata': {...}},
        ...
    ]
    
    Args:
        paper: QASPER paper dictionary
    
    Returns:
        List of text chunks
    """
    chunks = []
    
    # Add title as first chunk
    if paper.get('title'):
        chunks.append({
            'type': 'text',
            'content': f"Title: {paper['title']}",
            'metadata': {
                'section': 'title',
                'page': 1
            }
        })
    
    # Add abstract
    if paper.get('abstract'):
        chunks.append({
            'type': 'text',
            'content': f"Abstract: {paper['abstract']}",
            'metadata': {
                'section': 'abstract',
                'page': 1
            }
        })
    
    # Add full text sections
    if paper.get('full_text'):
        page_num = 2
        for section_name, paragraphs in paper['full_text'].items():
            # Flatten and combine paragraphs in each section
            # QASPER paragraphs can be nested lists
            flat_paragraphs = []
            
            if isinstance(paragraphs, list):
                for para in paragraphs:
                    if isinstance(para, list):
                        # Nested list - flatten it
                        flat_paragraphs.extend([str(p) for p in para if p])
                    elif isinstance(para, str):
                        flat_paragraphs.append(para)
                    else:
                        flat_paragraphs.append(str(para))
                
                section_text = f"{section_name}\n\n" + "\n\n".join(flat_paragraphs)
            else:
                section_text = f"{section_name}\n\n{str(paragraphs)}"
            
            # Only add if there's actual content
            if section_text.strip():
                chunks.append({
                    'type': 'text',
                    'content': section_text,
                    'metadata': {
                        'section': section_name,
                        'page': page_num
                    }
                })
                page_num += 1
    
    return chunks


def initialize_qasper_index():
    """
    Initialize separate ChromaDB collection for QASPER evaluation.
    This keeps evaluation separate from regular usage.
    """
    global _qasper_client, _qasper_collection, _qasper_docstore, _qasper_embedding_model
    
    if _qasper_collection is not None:
        return _qasper_collection
    
    # Create evaluation-specific directory
    eval_chroma_path = Path(__file__).parent.parent / "chroma_qasper_eval"
    eval_chroma_path.mkdir(exist_ok=True)
    
    print("Initializing QASPER evaluation index...")
    
    # Initialize ChromaDB client
    _qasper_client = chromadb.PersistentClient(
        path=str(eval_chroma_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    
    # Get or create collection
    _qasper_collection = _qasper_client.get_or_create_collection(
        name="qasper_evaluation",
        metadata={"description": "QASPER papers for evaluation"}
    )
    
    # Initialize embedding model
    _qasper_embedding_model = get_embedder()
    _qasper_docstore = {}
    
    print(f"QASPER index initialized at: {eval_chroma_path}")
    return _qasper_collection


def index_qasper_papers(papers_data, paper_ids: Optional[List[str]] = None, 
                        use_summarization: bool = False):
    """
    Index QASPER papers into ChromaDB for evaluation.
    
    Args:
        papers_data: QASPER dataset papers
        paper_ids: Optional list of paper IDs (defaults to indices)
        use_summarization: Whether to summarize long chunks (slower but may improve quality)
    
    Returns:
        Number of papers indexed
    """
    global _qasper_docstore
    
    initialize_qasper_index()
    
    if paper_ids is None:
        paper_ids = [str(i) for i in range(len(papers_data))]
    
    print(f"\nIndexing {len(papers_data)} papers into QASPER evaluation index...")
    
    if use_summarization:
        summarizer = get_summarizer()
        print("Summarization enabled (this will take longer)")
    
    indexed_count = 0
    
    for idx, paper in enumerate(tqdm(papers_data, desc="Indexing papers")):
        paper_id = paper_ids[idx]
        
        # Convert paper to chunks
        chunks = convert_qasper_paper_to_chunks(paper)
        
        if not chunks:
            continue
        
        # Process each chunk
        for chunk_idx, chunk in enumerate(chunks):
            content = chunk['content']
            
            # Optional summarization for very long chunks
            if use_summarization and len(content) > 1000:
                summary = summarizer(
                    content[:1024],  # Truncate to model limit
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                embed_text = f"summary: {summary}"
            else:
                embed_text = content[:512]  # Use first 512 chars for embedding
            
            # Generate embedding
            embedding = _qasper_embedding_model.encode(
                embed_text,
                convert_to_tensor=False,
                # convert_to_numpy=True, ?
                show_progress_bar=False
            ).tolist()
            
            # Create unique ID
            chunk_id = f"{paper_id}_chunk_{chunk_idx}"
            
            # Store in ChromaDB
            _qasper_collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[{
                    'paper_id': paper_id,
                    'chunk_index': chunk_idx,
                    'section': chunk['metadata'].get('section', 'unknown'),
                    'page': chunk['metadata'].get('page', 0)
                }]
            )
            
            # Store in docstore
            _qasper_docstore[chunk_id] = {
                'content': content,
                'metadata': chunk['metadata'],
                'paper_id': paper_id
            }
        
        indexed_count += 1
    
    print(f"\nSuccessfully indexed {indexed_count} papers")
    print(f"Total chunks in index: {len(_qasper_docstore)}")
    
    return indexed_count


def query_qasper_rag(question: str, paper_id: Optional[str] = None, 
                     apply_filtering: bool = True,
                     use_hybrid_qa: bool = True) -> Tuple[str, List[str]]:
    """
    Query the QASPER index using hybrid extractive+generative approach.
    
    Strategy:
    1. Retrieve and filter relevant chunks
    2. Try extractive QA first (RoBERTa) - fast and gets exact spans
    3. If extractive answer is good, use it (better F1 overlap with references)
    4. Otherwise, use generative QA (Llama) for more complex questions
    
    Args:
        question: Question to answer
        paper_id: Optional paper ID to restrict search
        apply_filtering: Whether to apply two-stage filtering
        use_hybrid_qa: Whether to use hybrid extractive+generative approach
    
    Returns:
        (answer, context_chunks) tuple
    """
    global _qasper_collection, _qasper_embedding_model, _qasper_docstore
    
    if _qasper_collection is None:
        raise RuntimeError("QASPER index not initialized. Call index_qasper_papers() first.")
    
    # Generate question embedding
    question_embedding = _qasper_embedding_model.encode(
        question,
        convert_to_tensor=False,
        show_progress_bar=False
    ).tolist()
    
    # Query ChromaDB
    k = INITIAL_RETRIEVAL_K if apply_filtering else 5
    query_kwargs = {
        'query_embeddings': [question_embedding],
        'n_results': k,
        'include': ['documents', 'distances', 'metadatas']
    }
    
    # Filter by paper_id if specified
    if paper_id:
        query_kwargs['where'] = {'paper_id': paper_id}
    
    results = _qasper_collection.query(**query_kwargs)
    
    if not results['documents'][0]:
        return "No relevant information found.", []
    
    # Extract results
    documents = results['documents'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    
    # Stage 1: Similarity filtering (if enabled)
    if apply_filtering:
        from sentence_transformers import util
        
        filtered_docs = []
        filtered_meta = []
        
        for doc, dist, meta in zip(documents, distances, metadatas):
            # Convert distance to similarity (cosine)
            similarity = 1 - dist
            
            if similarity >= SIMILARITY_THRESHOLD:
                filtered_docs.append(doc)
                filtered_meta.append(meta)
        
        if not filtered_docs:
            # No docs passed filtering, use top 5 anyway
            filtered_docs = documents[:5]
            filtered_meta = metadatas[:5]
        
        # Stage 2: Cross-encoder reranking
        reranker = get_reranker()
        
        pairs = [[question, doc] for doc in filtered_docs]
        scores = reranker.predict(pairs, show_progress_bar=False)
        
        # Sort by score and take top K
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_k_indices = ranked_indices[:RERANKER_TOP_K]
        
        context_chunks = [filtered_docs[i] for i in top_k_indices]
    else:
        # No filtering, use top 5
        context_chunks = documents[:5]
    
    # Combine context
    context = "\n\n".join(context_chunks)
    clean_context = context.replace("summary:", "").replace("summarize:", "").strip()
    
    # HYBRID APPROACH: Try extractive QA first
    if use_hybrid_qa:
        try:
            extractive_qa = get_qa_pipeline()
            
            # Use limited context for extractive QA (model limit)
            extractive_context = clean_context[:2000]
            
            extractive_result = extractive_qa(
                question=question,
                context=extractive_context
            )
            
            extractive_answer = extractive_result['answer'].strip()
            extractive_score = extractive_result['score']
            
            # Check if extractive answer is good enough
            # Good extractive answers: high confidence + substantial length
            answer_words = extractive_answer.split()
            is_good_extractive = (
                extractive_score > 0.5 and  # High confidence threshold (was 0.2, too permissive)
                len(answer_words) >= 15 and  # Substantial length required (was 4, too short)
                len(answer_words) <= 100    # Not unreasonably long
            )
            
            if is_good_extractive:
                # Extractive answer is good, use it directly
                return extractive_answer, context_chunks
            
            # Otherwise, fall through to generative QA
        
        except Exception as e:
            # If extractive QA fails, fall back to generative
            pass
    
    # GENERATIVE APPROACH: Use Llama for complex questions or when extractive fails
    gen_qa = get_generative_qa()
    
    # Optimized prompt: encourage comprehensive, detailed answers
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an AI assistant analyzing research papers. Provide thorough, detailed, and comprehensive answers to questions. Your answers should be substantial (around 100-120 words) and fully address all aspects of the question using information from the context.<|eot_id|><|start_header_id|>user<|end_header_id|>

Context from research paper:
{clean_context[:CONTEXT_MAX_CHARS]}

Question: {question}

Provide a detailed, comprehensive answer to the question using information from the context. Your answer should be thorough (around 100-120 words) and include all relevant information that addresses the question. In case of a Yes/No answer, please restrict the answer to a single word: "Yes" or "No".<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    result = gen_qa(
        prompt,
        max_new_tokens=ANSWER_MAX_LENGTH,
        min_new_tokens=ANSWER_MIN_LENGTH,
        do_sample=True,
        temperature=0.7,  # Balanced temperature for comprehensive yet focused answers
        top_p=0.9,
        repetition_penalty=1.1,
        return_full_text=False
    )
    
    answer = result[0]["generated_text"].strip()
    
    # Post-process: Remove common artifacts
    if answer.startswith("Answer:"):
        answer = answer[7:].strip()
    if answer.startswith("Based on the context,"):
        answer = answer[21:].strip()
    
    return answer, context_chunks


def evaluate_on_qasper(papers_data, num_questions: Optional[int] = None,
                       apply_filtering: bool = True,
                       use_hybrid_qa: bool = True,
                       save_predictions: bool = True,
                       output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """
    Run full evaluation on QASPER dataset.
    
    Args:
        papers_data: QASPER dataset papers
        num_questions: Max number of questions to evaluate (None = all)
        apply_filtering: Whether to use two-stage filtering
        use_hybrid_qa: Whether to use hybrid extractive+generative QA
        save_predictions: Whether to save predictions to file
        output_dir: Directory to save results
    
    Returns:
        Dictionary with evaluation metrics and results
    """
    print("\n" + "="*80)
    print("STARTING QASPER EVALUATION")
    print("="*80)
    
    # Index papers if not already indexed
    if _qasper_collection is None or _qasper_collection.count() == 0:
        index_qasper_papers(papers_data)
    
    # Collect all questions
    all_questions = []
    for paper_idx, paper in enumerate(papers_data):
        paper_id = str(paper_idx)
        
        if 'qas' not in paper:
            continue
        
        qas_data = paper['qas']
        
        # QASPER structure: qas is a dict with parallel lists
        # qas['question'] = [q1, q2, q3, ...]
        # qas['answers'] = [ans1, ans2, ans3, ...]
        if isinstance(qas_data, dict):
            questions_list = qas_data.get('question', [])
            answers_list = qas_data.get('answers', [])
            
            # Ensure both are lists
            if not isinstance(questions_list, list):
                questions_list = [questions_list] if questions_list else []
            if not isinstance(answers_list, list):
                answers_list = [answers_list] if answers_list else []
            
            # Zip questions with their corresponding answers
            for i, question_text in enumerate(questions_list):
                if i >= len(answers_list):
                    break
                
                # Get answers for this question
                answer_set = answers_list[i]
                
                # Extract answer texts from the answer set
                answer_texts = []
                
                if isinstance(answer_set, dict) and 'answer' in answer_set:
                    # Answer set contains list of annotator answers
                    annotator_answers = answer_set['answer']
                    
                    if isinstance(annotator_answers, list):
                        for ann_ans in annotator_answers:
                            if not isinstance(ann_ans, dict):
                                continue
                            
                            # Try free_form_answer first
                            free_form = ann_ans.get('free_form_answer', '').strip()
                            if free_form:
                                answer_texts.append(free_form)
                            else:
                                # Use evidence if free_form is empty
                                evidence = ann_ans.get('evidence', [])
                                if evidence and isinstance(evidence, list):
                                    # Join evidence sentences
                                    evidence_text = ' '.join([e for e in evidence if isinstance(e, str)])
                                    if evidence_text.strip():
                                        answer_texts.append(evidence_text.strip())
                                elif isinstance(evidence, str) and evidence.strip():
                                    answer_texts.append(evidence.strip())
                                else:
                                    # Last resort: use extractive_spans
                                    spans = ann_ans.get('extractive_spans', [])
                                    if spans and isinstance(spans, list):
                                        spans_text = ', '.join([s for s in spans if isinstance(s, str)])
                                        if spans_text.strip():
                                            answer_texts.append(spans_text.strip())
                    
                    # Remove duplicates while preserving order
                    seen = set()
                    answer_texts = [x for x in answer_texts if not (x in seen or seen.add(x))]
                
                if question_text and answer_texts:
                    all_questions.append({
                        'paper_id': paper_id,
                        'question': question_text,
                        'reference_answers': answer_texts,
                        'paper_title': paper.get('title', 'Unknown')
                    })
    
    print(f"Total questions available: {len(all_questions)}")
    
    if num_questions:
        all_questions = all_questions[:num_questions]
        print(f"Evaluating on first {num_questions} questions")
    
    # Run evaluation
    predictions = []
    all_metrics = []
    
    print("\nGenerating answers...")
    
    for qa_item in tqdm(all_questions, desc="Answering questions"):
        try:
            # Generate answer
            predicted_answer, context = query_qasper_rag(
                qa_item['question'],
                paper_id=qa_item['paper_id'],
                apply_filtering=apply_filtering,
                use_hybrid_qa=use_hybrid_qa
            )
            
            # Evaluate against reference answers
            # Use first reference answer as primary
            reference_answer = qa_item['reference_answers'][0]
            
            metrics = evaluate_answer(predicted_answer, reference_answer)
            
            predictions.append({
                'paper_id': qa_item['paper_id'],
                'paper_title': qa_item['paper_title'],
                'question': qa_item['question'],
                'predicted_answer': predicted_answer,
                'reference_answers': qa_item['reference_answers'],
                'metrics': metrics
            })
            
            all_metrics.append(metrics)
        
        except Exception as e:
            print(f"\nError processing question: {e}")
            continue
    
    batch_metrics = evaluate_batch([p["predicted_answer"] for p in predictions], [p["reference_answers"][0] for p in predictions])
    assert batch_metrics["num_examples"] == len(predictions), "Mismatch in number of evaluated examples"
    
    # Calculate average metrics
    avg_metrics = {}
    if all_metrics:
        metric_keys = all_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0

    avg_metrics.update(batch_metrics["average_metrics"])  # Include batch evaluation metrics

    results = {
        'num_questions': len(predictions),
        'apply_filtering': apply_filtering,
        'use_hybrid_qa': use_hybrid_qa,
        'average_metrics': avg_metrics,
        'predictions': predictions
    }
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Questions evaluated: {len(predictions)}")
    print(f"Two-stage filtering: {'Enabled' if apply_filtering else 'Disabled'}")
    print(f"Hybrid extractive+generative QA: {'Enabled' if use_hybrid_qa else 'Disabled'}")
    print("\nAverage Metrics:")

    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save predictions
    if save_predictions:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        filtering_suffix = "with_filtering" if apply_filtering else "no_filtering"
        hybrid_suffix = "_hybrid" if use_hybrid_qa else "_generative_only"
        pred_file = output_path / f"qasper_predictions_{filtering_suffix}{hybrid_suffix}.json"
        
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nPredictions saved to: {pred_file}")
    
    return results


def clear_qasper_index():
    """
    Clear the QASPER evaluation index.
    """
    global _qasper_client, _qasper_collection, _qasper_docstore
    
    if _qasper_client:
        try:
            _qasper_client.delete_collection("qasper_evaluation")
            print("QASPER evaluation index cleared")
        except:
            pass
    
    _qasper_collection = None
    _qasper_docstore = {}


def compare_configurations(papers_data, num_questions: int = 50) -> Dict[str, Any]:
    """
    Compare RAG performance with and without two-stage filtering.
    
    Args:
        papers_data: QASPER dataset papers
        num_questions: Number of questions to test on
    
    Returns:
        Comparison results dictionary
    """
    print("\n" + "="*80)
    print("COMPARING CONFIGURATIONS")
    print("="*80)
    
    # Evaluate without filtering
    print("\n[1/2] Evaluating WITHOUT two-stage filtering...")
    results_no_filter = evaluate_on_qasper(
        papers_data,
        num_questions=num_questions,
        apply_filtering=False,
        save_predictions=True
    )
    
    # Evaluate with filtering
    print("\n[2/2] Evaluating WITH two-stage filtering...")
    results_with_filter = evaluate_on_qasper(
        papers_data,
        num_questions=num_questions,
        apply_filtering=True,
        save_predictions=True
    )
    
    # Calculate improvements
    comparison = {
        'no_filtering': results_no_filter['average_metrics'],
        'with_filtering': results_with_filter['average_metrics'],
        'improvements': {}
    }
    
    for metric in results_no_filter['average_metrics']:
        baseline = results_no_filter['average_metrics'][metric]
        improved = results_with_filter['average_metrics'][metric]
        
        if baseline > 0:
            pct_improvement = ((improved - baseline) / baseline) * 100
        else:
            pct_improvement = 0.0
        
        comparison['improvements'][metric] = {
            'absolute': improved - baseline,
            'percentage': pct_improvement
        }
    
    # Print comparison
    print("\n" + "="*80)
    print("CONFIGURATION COMPARISON")
    print("="*80)
    print(f"\nMetric                  | No Filtering | With Filtering | Improvement")
    print("-" * 80)
    
    for metric in comparison['no_filtering']:
        no_filt = comparison['no_filtering'][metric]
        with_filt = comparison['with_filtering'][metric]
        improvement = comparison['improvements'][metric]['percentage']
        
        print(f"{metric:22s} | {no_filt:11.4f} | {with_filt:14.4f} | {improvement:+.2f}%")
    
    return comparison


if __name__ == "__main__":
    # Quick test
    print("Loading QASPER dataset...")
    papers = load_qasper_dataset(split="validation")
    
    print("\nIndexing papers...")
    index_qasper_papers(papers)
    
    print("\nTesting query...")
    answer, context = query_qasper_rag("What is the main contribution?", paper_id="0")
    print(f"\nAnswer: {answer}")
    print(f"\nUsed {len(context)} context chunks")

    # Evaluation with optimized settings (2-stage filtering + hybrid QA)
    print("="*80)
    print("RUNNING EVALUATION WITH OPTIMIZED HYBRID QA SETTINGS")
    print("="*80)
    print("Settings:")
    print("  - Initial retrieval: 20 chunks")
    print("  - Similarity threshold: 0.2")
    print("  - Reranker top-K: 6 chunks")
    print("  - Context window: 3000 chars")
    print("  - Hybrid QA: Extractive (RoBERTa) + Generative (Llama)")
    print("="*80)

    # Run evaluation on first 50 questions (adjust as needed)
    results_optimized = evaluate_on_qasper(
        papers,
        num_questions=50,
        apply_filtering=True,
        use_hybrid_qa=True,  # GENERATIVE ONLY (like old system) or HYBRID (extractive+generative)
        save_predictions=True,
        output_dir=f"evaluation_results_baseline"
    )

    print("\n" + "="*80)
    print("OPTIMIZED RESULTS SUMMARY")
    print("="*80)
    for metric, value in results_optimized['average_metrics'].items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*80)
    
    # Baseline Evaluation (Generative-only)
    print("="*80)
    print("RUNNING BASELINE EVALUATION (GENERATIVE-ONLY, NO FILTERING)")
    print("="*80)
    print("This uses only Llama (no extractive QA) for comparison")
    print("="*80)

    results_baseline = evaluate_on_qasper(
        papers,
        num_questions=50,
        apply_filtering=True,
        use_hybrid_qa=False,  # GENERATIVE ONLY (like old system)
        save_predictions=True,
        output_dir=f"evaluation_results_baseline"
    )

    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)
    for metric, value in results_baseline['average_metrics'].items():
        print(f"{metric:20s}: {value:.4f}")
    print("="*80)

    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: OPTIMIZED vs BASELINE")
    print("="*80)
    print(f"{'Metric':<20} | {'Baseline':>12} | {'Optimized':>12} | {'Improvement':>12}")
    print("-"*80)

    for metric in results_baseline['average_metrics'].keys():
        baseline_val = results_baseline['average_metrics'][metric]
        optimized_val = results_optimized['average_metrics'][metric]

        if baseline_val > 0:
            improvement = ((optimized_val - baseline_val) / baseline_val) * 100
        else:
            improvement = 0.0

        print(f"{metric:<20} | {baseline_val:>12.4f} | {optimized_val:>12.4f} | {improvement:>+11.2f}%")

    print("="*80)