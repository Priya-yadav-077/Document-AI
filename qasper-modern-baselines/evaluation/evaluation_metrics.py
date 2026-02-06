"""
Evaluation Metrics Module

Calculates various metrics for comparing predicted answers to reference answers:
- F1 Score (token-level overlap)
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- BERTScore (semantic similarity using BERT embeddings)
"""

import re
import string
from typing import Dict, List, Any
from collections import Counter
import numpy as np


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for fair comparison.
    
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    
    Args:
        text: Answer text to normalize
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def get_tokens(text: str) -> List[str]:
    """
    Tokenize text into words.
    
    Args:
        text: Text to tokenize
    
    Returns:
        List of tokens
    """
    normalized = normalize_answer(text)
    return normalized.split()


def calculate_f1_score(prediction: str, reference: str) -> float:
    """
    Calculate F1 score between prediction and reference.
    
    F1 = 2 * (precision * recall) / (precision + recall)
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0
    
    # Calculate token overlap
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def calculate_exact_match(prediction: str, reference: str) -> float:
    """
    Calculate exact match score (1.0 if exact match after normalization, else 0.0).
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    pred_normalized = normalize_answer(prediction)
    ref_normalized = normalize_answer(reference)
    
    return 1.0 if pred_normalized == ref_normalized else 0.0


def calculate_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    ROUGE-1: Unigram overlap
    ROUGE-2: Bigram overlap
    ROUGE-L: Longest common subsequence
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        Dictionary with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    except ImportError:
        print("Warning: rouge_score not installed. Install with: pip install rouge-score")
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }


def calculate_bert_score(predictions: List[str], references: List[str],
                        device: str = 'cuda', batch_size: int = 32) -> Dict[str, float]:
    """
    Calculate BERTScore for a batch of predictions.
    
    BERTScore measures semantic similarity using contextual embeddings.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        from bert_score import score
        
        # Calculate BERTScore
        P, R, F1 = score(
            predictions,
            references,
            lang='en',
            device=device,
            batch_size=batch_size,
            verbose=False
        )
        
        return {
            'bertscore_precision': P.mean().item(),
            'bertscore_recall': R.mean().item(),
            'bertscore_f1': F1.mean().item()
        }
    
    except ImportError:
        print("Warning: bert_score not installed. Install with: pip install bert-score")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }
    except Exception as e:
        print(f"Warning: BERTScore calculation failed: {e}")
        return {
            'bertscore_precision': 0.0,
            'bertscore_recall': 0.0,
            'bertscore_f1': 0.0
        }


def evaluate_answer(prediction: str, reference: str) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single answer.
    
    Calculates:
    - F1 score
    - Exact Match
    - ROUGE-1, ROUGE-2, ROUGE-L
    
    Note: BERTScore is calculated separately in batch for efficiency.
    
    Args:
        prediction: Predicted answer
        reference: Reference answer
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # F1 score
    metrics['f1'] = calculate_f1_score(prediction, reference)
    
    # Exact Match
    metrics['exact_match'] = calculate_exact_match(prediction, reference)
    
    # ROUGE scores
    rouge_scores = calculate_rouge_scores(prediction, reference)
    metrics.update(rouge_scores)
    
    # Answer length (for analysis)
    metrics['pred_length'] = len(get_tokens(prediction))
    metrics['ref_length'] = len(get_tokens(reference))
    
    return metrics


def evaluate_batch(predictions: List[str], references: List[str],
                   calculate_bertscore: bool = True,
                   device: str = 'cuda') -> Dict[str, Any]:
    """
    Evaluate a batch of predictions against references.
    
    Args:
        predictions: List of predicted answers
        references: List of reference answers
        calculate_bertscore: Whether to calculate BERTScore (slower)
        device: Device for BERTScore computation
    
    Returns:
        Dictionary with individual and average metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions must match number of references")
    
    individual_metrics = []
    
    # Calculate per-example metrics
    for pred, ref in zip(predictions, references):
        metrics = evaluate_answer(pred, ref)
        individual_metrics.append(metrics)
    
    # Calculate average metrics
    avg_metrics = {}
    if individual_metrics:
        metric_keys = individual_metrics[0].keys()
        for key in metric_keys:
            values = [m[key] for m in individual_metrics]
            avg_metrics[key] = sum(values) / len(values)
    
    # Calculate BERTScore for entire batch (more efficient)
    if calculate_bertscore:
        bertscore_metrics = calculate_bert_score(predictions, references, device=device)
        avg_metrics.update(bertscore_metrics)
    
    return {
        'individual_metrics': individual_metrics,
        'average_metrics': avg_metrics,
        'num_examples': len(predictions)
    }


def generate_metrics_report(results: Dict[str, Any], 
                           output_file: str = None) -> str:
    """
    Generate a formatted metrics report.
    
    Args:
        results: Results dictionary from evaluate_batch or evaluate_on_qasper
        output_file: Optional file to save report to
    
    Returns:
        Formatted report string
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("EVALUATION METRICS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    if 'num_questions' in results:
        report_lines.append(f"Total Questions Evaluated: {results['num_questions']}")
    elif 'num_examples' in results:
        report_lines.append(f"Total Examples Evaluated: {results['num_examples']}")
    
    if 'apply_filtering' in results:
        report_lines.append(f"Two-Stage Filtering: {'Enabled' if results['apply_filtering'] else 'Disabled'}")
    
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("AVERAGE METRICS")
    report_lines.append("-" * 80)
    
    # Get metrics
    avg_metrics = results.get('average_metrics', {})
    
    # Group metrics
    core_metrics = ['f1', 'exact_match']
    rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
    bert_metrics = ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']
    length_metrics = ['pred_length', 'ref_length']
    
    # Core metrics
    report_lines.append("\nCore Metrics:")
    for metric in core_metrics:
        if metric in avg_metrics:
            report_lines.append(f"  {metric:20s}: {avg_metrics[metric]:.4f}")
    
    # ROUGE metrics
    report_lines.append("\nROUGE Scores:")
    for metric in rouge_metrics:
        if metric in avg_metrics:
            report_lines.append(f"  {metric:20s}: {avg_metrics[metric]:.4f}")
    
    # BERTScore metrics
    if any(m in avg_metrics for m in bert_metrics):
        report_lines.append("\nBERTScore:")
        for metric in bert_metrics:
            if metric in avg_metrics:
                report_lines.append(f"  {metric:20s}: {avg_metrics[metric]:.4f}")
    
    # Length statistics
    if any(m in avg_metrics for m in length_metrics):
        report_lines.append("\nAnswer Length (tokens):")
        for metric in length_metrics:
            if metric in avg_metrics:
                report_lines.append(f"  {metric:20s}: {avg_metrics[metric]:.2f}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    
    return report


def compare_two_systems(results1: Dict[str, Any], results2: Dict[str, Any],
                       system1_name: str = "System 1",
                       system2_name: str = "System 2") -> str:
    """
    Generate a comparison report between two systems.
    
    Args:
        results1: Results from first system
        results2: Results from second system
        system1_name: Name of first system
        system2_name: Name of second system
    
    Returns:
        Formatted comparison report
    """
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("SYSTEM COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    metrics1 = results1.get('average_metrics', {})
    metrics2 = results2.get('average_metrics', {})
    
    # Header
    report_lines.append(f"{'Metric':<25} | {system1_name:>15} | {system2_name:>15} | {'Difference':>12}")
    report_lines.append("-" * 80)
    
    # Compare each metric
    all_metrics = set(metrics1.keys()) | set(metrics2.keys())
    
    for metric in sorted(all_metrics):
        val1 = metrics1.get(metric, 0.0)
        val2 = metrics2.get(metric, 0.0)
        diff = val2 - val1
        
        # Format difference with + or -
        diff_str = f"{diff:+.4f}"
        if diff > 0:
            diff_str += " (better)"
        elif diff < 0:
            diff_str += " (worse)"
        
        report_lines.append(f"{metric:<25} | {val1:>15.4f} | {val2:>15.4f} | {diff_str:>12}")
    
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Test metrics
    pred = "The main contribution is a novel attention mechanism for transformers."
    ref = "The paper proposes a new attention mechanism that improves transformer efficiency."
    
    print("Testing evaluation metrics...")
    print(f"\nPrediction: {pred}")
    print(f"Reference: {ref}")
    
    metrics = evaluate_answer(pred, ref)
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
