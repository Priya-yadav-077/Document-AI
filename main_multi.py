# main_multi.py - Multi-PDF comparison mode entry point
import argparse
import sys
from multi_pdf_pipeline import index_paper, query_single_paper, reset_multi_index
from answer_comparator import compare_answers
from config import COMPARISON_METHOD

def main():
    parser = argparse.ArgumentParser(
        description="Multi-PDF RAG with Answer Comparison (Research Paper Analysis)"
    )
    
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset multi-PDF index (clear all papers)"
    )
    
    parser.add_argument(
        "--index",
        nargs='+',
        metavar='ARGS',
        help="Index a paper: --index <pdf_path> <paper_id> <paper_title>"
    )
    
    parser.add_argument(
        "--compare-query",
        type=str,
        metavar="QUESTION",
        help="Ask a question and compare answers from both papers"
    )
    
    parser.add_argument(
        "--paper1-id",
        type=str,
        default="paper1",
        help="Paper 1 identifier (default: paper1)"
    )
    
    parser.add_argument(
        "--paper2-id",
        type=str,
        default="paper2",
        help="Paper 2 identifier (default: paper2)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        choices=["llama_judge", "similarity", "hybrid"],
        default=COMPARISON_METHOD,
        help="Comparison method (default: llama_judge)"
    )
    
    args = parser.parse_args()
    
    # Reset index
    if args.reset:
        print("Resetting multi-PDF index...")
        reset_multi_index()
        print("Index reset complete.")
        return
    
    # Index a paper
    if args.index:
        if len(args.index) < 3:
            print("Error: --index requires 3 arguments: <pdf_path> <paper_id> <paper_title>")
            print("Example: --index paper1.pdf paper1 'Attention Is All You Need'")
            sys.exit(1)
        
        pdf_path = args.index[0]
        paper_id = args.index[1]
        paper_title = ' '.join(args.index[2:])  # Allow multi-word titles
        
        print(f"\n{'='*80}")
        print(f"INDEXING PAPER")
        print(f"{'='*80}")
        print(f"PDF: {pdf_path}")
        print(f"ID: {paper_id}")
        print(f"Title: {paper_title}")
        print(f"{'='*80}\n")
        
        try:
            index_paper(pdf_path, paper_id, paper_title)
            print(f"\n✓ Successfully indexed {paper_id}: {paper_title}")
        except Exception as e:
            print(f"\n✗ Error indexing paper: {e}")
            sys.exit(1)
        
        return
    
    # Compare query
    if args.compare_query:
        question = args.compare_query
        paper1_id = args.paper1_id
        paper2_id = args.paper2_id
        
        print(f"\n{'='*80}")
        print(f"MULTI-PAPER QUERY")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Papers: {paper1_id} vs {paper2_id}")
        print(f"Method: {args.method}")
        print(f"{'='*80}\n")
        
        # Query each paper separately
        try:
            print(f"\n{'#'*80}")
            print(f"# QUERYING {paper1_id.upper()}")
            print(f"{'#'*80}\n")
            answer1 = query_single_paper(question, paper1_id)
            
            print(f"\n{'#'*80}")
            print(f"# QUERYING {paper2_id.upper()}")
            print(f"{'#'*80}\n")
            answer2 = query_single_paper(question, paper2_id)
            
        except Exception as e:
            print(f"\n✗ Error querying papers: {e}")
            print("\nMake sure both papers are indexed first using --index")
            sys.exit(1)
        
        # Display individual answers
        print(f"\n{'='*80}")
        print(f"ANSWER FROM {answer1['paper_title'].upper()}")
        print(f"{'='*80}")
        print(answer1['response'])
        print(f"\nChunks used: {len(answer1['context']['texts'])}")
        
        print(f"\n{'='*80}")
        print(f"ANSWER FROM {answer2['paper_title'].upper()}")
        print(f"{'='*80}")
        print(answer2['response'])
        print(f"\nChunks used: {len(answer2['context']['texts'])}")
        
        # Compare answers
        print(f"\n{'='*80}")
        print(f"LLAMA JUDGE COMPARISON")
        print(f"{'='*80}\n")
        
        comparison = compare_answers(question, [answer1, answer2], method=args.method)
        
        print(comparison['comparison'])
        
        print(f"\n{'='*80}")
        print(f"FINAL DECISION")
        print(f"{'='*80}")
        print(f"Winner: {comparison['winner_title']}")
        print(f"{'='*80}\n")
        
        return
    
    # No arguments provided
    parser.print_help()

if __name__ == "__main__":
    main()
