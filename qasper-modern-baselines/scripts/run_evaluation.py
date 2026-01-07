"""
Main evaluation script for running modern LLM baselines on Qasper
Replicates the evaluation protocol from the LED baseline paper
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_qasper_data
from baselines.gemini_baseline import GeminiBaseline
from baselines.llama_baseline import LlamaBaseline
from evaluation.metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate modern LLM baselines on Qasper dataset"
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'dev', 'test'],
        default='dev',
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=[
            # Gemini models
            'gemini-2.5-pro', 'gemini-2.5-flash', 
            'gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash',
            # LLaMA models via Groq
            'llama-3.3-70b', 'llama-3.1-70b', 'llama-3.1-8b',
            'llama-4-maverick-17b', 'llama-4-scout-17b',
            # Special
            'all'
        ],
        default='gemini-2.5-flash',
        help='Model to evaluate (or "all" for all models)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of papers to evaluate (None = all)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing Qasper data files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--rate_limit',
        type=float,
        default=1.0,
        help='Delay between API calls (seconds)'
    )
    parser.add_argument(
        '--use_evidence_prompt',
        action='store_true',
        help='Use evidence-based prompting'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling papers (default: 42 for reproducibility)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct data path
    data_file = f"qasper-{args.split}-v0.3.json"
    data_path = os.path.join(args.data_dir, data_file)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found: {data_path}")
        print(f"   Run: bash scripts/download_data.sh")
        return
    
    # Load data
    print(f"\n{'='*80}")
    print(f"QASPER EVALUATION - {args.split.upper()} SET")
    print(f"{'='*80}\n")
    
    print(f"Loading data from {data_path}...")
    
    # Use random sampling if num_samples specified, otherwise load all
    random_seed = args.seed if args.num_samples else None
    papers = load_qasper_data(data_path, max_samples=args.num_samples, random_seed=random_seed)
    
    total_questions = sum(len(p['questions']) for p in papers)
    
    # Print question type distribution
    question_types = {}
    for paper in papers:
        for q in paper['questions']:
            q_type = q['question_type']
            question_types[q_type] = question_types.get(q_type, 0) + 1
    
    print(f"Loaded {len(papers)} papers with {total_questions} questions")
    print(f"Question type distribution: {question_types}\n")
    
    # Determine which models to run
    if args.model == 'all':
        models_to_run = ['gemini-2.5-flash', 'gemini-2.5-pro', 'llama-3.3-70b', 'llama-3.1-8b']
    else:
        models_to_run = [args.model]
    
    # Run evaluation for each model
    all_results = {}
    
    for model_name in models_to_run:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*80}\n")
        
        # Initialize baseline
        if 'gemini' in model_name:
            baseline = GeminiBaseline(
                model=model_name,
                use_evidence_prompt=args.use_evidence_prompt
            )
        elif 'llama' in model_name:
            baseline = LlamaBaseline(
                model=model_name,  # Pass the model name directly, will be mapped internally
                use_evidence_prompt=args.use_evidence_prompt
            )
        else:
            print(f"Unknown model: {model_name}")
            continue
        
        # Print model info
        model_info = baseline.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Provider: {model_info['provider']}")
        ctx_window = model_info['context_window']
        if isinstance(ctx_window, (int, float)):
            print(f"Context Window: {ctx_window:,} tokens")
        else:
            print(f"Context Window: {ctx_window}")
        print(f"Temperature: {model_info['temperature']}")
        print(f"Evidence Prompting: {model_info['use_evidence_prompt']}\n")
        
        # Run predictions
        print(f"Running predictions...")
        start_time = datetime.now()
        
        results = baseline.answer_batch(
            papers,
            rate_limit_delay=args.rate_limit,
            verbose=True
        )
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        print(f"\n✅ Completed {len(results)} predictions in {elapsed:.1f}s")
        print(f"   Average: {elapsed/len(results):.2f}s per question")
        
        # Evaluate
        print(f"\nEvaluating metrics...")
        metrics = evaluate_predictions(results)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS: {model_name}")
        print(f"{'='*80}\n")
        
        print(f"Overall Metrics:")
        print(f"  Answer F1:     {metrics['answer_f1']*100:.2f}%")
        print(f"  Exact Match:   {metrics['exact_match']*100:.2f}%")
        print(f"  ROUGE-L:       {metrics['rouge_l']*100:.2f}%")
        print(f"  BERTScore F1:  {metrics['bert_score']['f1']*100:.2f}%")
        print(f"  Evidence F1:   {metrics['evidence_f1']*100:.2f}%")
        
        print(f"\nBreakdown by Question Type:")
        for q_type, type_metrics in metrics['by_question_type'].items():
            print(f"\n  {q_type}:")
            print(f"    Count:         {type_metrics['count']}")
            print(f"    Answer F1:     {type_metrics['answer_f1']*100:.2f}%")
            print(f"    Exact Match:   {type_metrics['exact_match']*100:.2f}%")
            print(f"    ROUGE-L:       {type_metrics['rouge_l']*100:.2f}%")
            print(f"    BERTScore F1:  {type_metrics['bert_score']['f1']*100:.2f}%")
            # Note: Evidence F1 not reported per type (not meaningful - evidence selection is question-type agnostic)
        
        # Save results
        output_file = os.path.join(
            args.output_dir,
            f"{model_name}_{args.split}_results.json"
        )
        
        results_to_save = {
            'model': model_name,
            'model_info': model_info,
            'split': args.split,
            'num_papers': len(papers),
            'num_questions': total_questions,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'predictions': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        all_results[model_name] = metrics
    
    # Print comparison if multiple models
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Model':<25} {'Answer F1':>12} {'Exact Match':>12} {'ROUGE-L':>12} {'BERTScore':>12} {'Evidence F1':>12}")
        print(f"{'-'*110}")
        
        for model_name, metrics in all_results.items():
            print(f"{model_name:<25} "
                  f"{metrics['answer_f1']*100:>11.2f}% "
                  f"{metrics['exact_match']*100:>11.2f}% "
                  f"{metrics['rouge_l']*100:>11.2f}% "
                  f"{metrics['bert_score']['f1']*100:>11.2f}% "
                  f"{metrics['evidence_f1']*100:>11.2f}%")
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
