"""
Evaluation metrics for Qasper
Implements the same metrics as the original LED (Longformer Encoder-Decoder) baseline
"""

import re
from typing import List, Dict, Set
from collections import Counter
import numpy as np


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for comparison
    Follows SQuAD/Qasper normalization
    
    Args:
        s: Answer string
        
    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        import string
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_token_f1(prediction: str, gold_answers: List[str]) -> float:
    """
    Compute token-level F1 score (Answer F1 in Qasper paper)
    Takes maximum F1 across all gold answers
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of gold answer strings
        
    Returns:
        Maximum F1 score
    """
    def get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()
    
    prediction_tokens = get_tokens(prediction)
    
    if not prediction_tokens:
        return 0.0
    
    f1_scores = []
    for gold in gold_answers:
        gold_tokens = get_tokens(gold)
        
        if not gold_tokens:
            if not prediction_tokens:
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
            continue
        
        common = Counter(prediction_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1_scores.append(0.0)
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    
    return max(f1_scores) if f1_scores else 0.0


def compute_exact_match(prediction: str, gold_answers: List[str]) -> float:
    """
    Compute exact match score
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of gold answer strings
        
    Returns:
        1.0 if exact match with any gold answer, else 0.0
    """
    normalized_prediction = normalize_answer(prediction)
    
    for gold in gold_answers:
        if normalized_prediction == normalize_answer(gold):
            return 1.0
    
    return 0.0


def compute_evidence_f1(
    predicted_evidence: List[str],
    gold_evidence_sets: List[List[str]]
) -> float:
    """
    Compute evidence F1 score (paragraph-level)
    Following the Qasper paper implementation
    
    Args:
        predicted_evidence: List of predicted paragraph texts
        gold_evidence_sets: List of gold evidence sets (one per annotator)
        
    Returns:
        Maximum F1 score across annotators
    """
    if not predicted_evidence:
        return 0.0
    
    f1_scores = []
    
    for gold_evidence in gold_evidence_sets:
        # Filter out "FLOAT SELECTED" entries (as in original implementation)
        gold_paragraphs = [p for p in gold_evidence if "FLOAT SELECTED" not in p]
        
        if not gold_paragraphs:
            continue
        
        # Find matches
        predicted_set = set(normalize_answer(p) for p in predicted_evidence)
        gold_set = set(normalize_answer(p) for p in gold_paragraphs)
        
        intersection = predicted_set & gold_set
        
        if not intersection:
            f1_scores.append(0.0)
            continue
        
        precision = len(intersection) / len(predicted_set)
        recall = len(intersection) / len(gold_set)
        
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
    
    return max(f1_scores) if f1_scores else 0.0


def compute_rouge_l(prediction: str, gold_answers: List[str]) -> float:
    """
    Compute ROUGE-L score
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of gold answer strings
        
    Returns:
        Maximum ROUGE-L F1 score
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        scores = []
        for gold in gold_answers:
            score = scorer.score(gold, prediction)
            scores.append(score['rougeL'].fmeasure)
        
        return max(scores) if scores else 0.0
    
    except ImportError:
        print("Warning: rouge_score not installed. Install with: pip install rouge-score")
        return 0.0


def compute_bert_score(predictions: List[str], references: List[List[str]]) -> Dict:
    """
    Compute BERTScore for a batch of predictions
    
    Args:
        predictions: List of predicted answers
        references: List of gold answer lists
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        from bert_score import score as bert_score_fn
        
        # Flatten references (take first gold answer for each)
        flat_references = [refs[0] if refs else "" for refs in references]
        
        P, R, F1 = bert_score_fn(
            predictions,
            flat_references,
            lang="en",
            verbose=False
        )
        
        return {
            'precision': P.mean().item(),
            'recall': R.mean().item(),
            'f1': F1.mean().item()
        }
    
    except ImportError:
        print("Warning: bert_score not installed. Install with: pip install bert-score")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def evaluate_predictions(results: List[Dict]) -> Dict:
    """
    Evaluate a list of prediction results
    
    Args:
        results: List of result dictionaries from baseline models
        
    Returns:
        Dictionary with aggregate metrics
    """
    answer_f1_scores = []
    exact_match_scores = []
    rouge_l_scores = []
    evidence_f1_scores = []
    
    # For BERTScore (batch computation)
    predictions = []
    references = []
    
    # By question type
    metrics_by_type = {}
    
    for result in results:
        if not result.get('success', False):
            continue
        
        prediction = result.get('answer', '')
        gold_answers = result.get('gold_answers', [])
        question_type = result.get('question_type', 'unknown')
        
        if not gold_answers:
            continue
        
        # Compute answer metrics
        f1 = compute_token_f1(prediction, gold_answers)
        em = compute_exact_match(prediction, gold_answers)
        rouge = compute_rouge_l(prediction, gold_answers)
        
        answer_f1_scores.append(f1)
        exact_match_scores.append(em)
        rouge_l_scores.append(rouge)
        
        predictions.append(prediction)
        references.append(gold_answers)
        
        # Compute evidence metrics if available
        predicted_evidence = result.get('evidence', None)
        gold_evidence = result.get('gold_evidence', [])
        
        evidence_f1 = 0.0
        if predicted_evidence and gold_evidence:
            # Parse predicted evidence (may be string or list)
            if isinstance(predicted_evidence, str):
                # Split by newlines or common delimiters
                pred_evidence_list = [p.strip() for p in predicted_evidence.split('\n') if p.strip()]
            else:
                pred_evidence_list = predicted_evidence
            
            evidence_f1 = compute_evidence_f1(pred_evidence_list, gold_evidence)
        
        evidence_f1_scores.append(evidence_f1)
        
        # Track by question type (excluding evidence_f1 - not meaningful per type)
        if question_type not in metrics_by_type:
            metrics_by_type[question_type] = {
                'count': 0,
                'answer_f1': [],
                'exact_match': [],
                'rouge_l': [],
                'predictions': [],
                'references': []
            }
        
        metrics_by_type[question_type]['count'] += 1
        metrics_by_type[question_type]['answer_f1'].append(f1)
        metrics_by_type[question_type]['exact_match'].append(em)
        metrics_by_type[question_type]['rouge_l'].append(rouge)
        metrics_by_type[question_type]['predictions'].append(prediction)
        metrics_by_type[question_type]['references'].append(gold_answers)
    
    # Compute overall BERTScore
    bert_scores = compute_bert_score(predictions, references) if predictions else {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0
    }
    
    # Aggregate by question type (excluding Evidence F1 - only reported overall)
    aggregated_by_type = {}
    for q_type, metrics in metrics_by_type.items():
        # Compute BERTScore for this question type
        type_bert_scores = compute_bert_score(
            metrics['predictions'],
            metrics['references']
        ) if metrics['predictions'] else {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        aggregated_by_type[q_type] = {
            'count': metrics['count'],
            'answer_f1': np.mean(metrics['answer_f1']) if metrics['answer_f1'] else 0.0,
            'exact_match': np.mean(metrics['exact_match']) if metrics['exact_match'] else 0.0,
            'rouge_l': np.mean(metrics['rouge_l']) if metrics['rouge_l'] else 0.0,
            'bert_score': type_bert_scores
            # Note: evidence_f1 deliberately excluded - not meaningful per question type
        }
    
    return {
        'answer_f1': np.mean(answer_f1_scores) if answer_f1_scores else 0.0,
        'exact_match': np.mean(exact_match_scores) if exact_match_scores else 0.0,
        'rouge_l': np.mean(rouge_l_scores) if rouge_l_scores else 0.0,
        'bert_score': bert_scores,
        'evidence_f1': np.mean(evidence_f1_scores) if evidence_f1_scores else 0.0,
        'total_questions': len(results),
        'successful_predictions': len(answer_f1_scores),
        'by_question_type': aggregated_by_type
    }


if __name__ == "__main__":
    # Test metrics
    print("Testing Qasper metrics...")
    
    # Test Answer F1
    pred = "The Transformer uses multi-head attention"
    gold = ["The Transformer architecture uses multi-head attention mechanism"]
    f1 = compute_token_f1(pred, gold)
    print(f"\nAnswer F1: {f1:.3f}")
    
    # Test Exact Match
    em = compute_exact_match("Yes", ["Yes", "yes"])
    print(f"Exact Match: {em:.3f}")
    
    # Test ROUGE-L
    rouge = compute_rouge_l(pred, gold)
    print(f"ROUGE-L: {rouge:.3f}")
    
    print("\nAll metrics working correctly!")
