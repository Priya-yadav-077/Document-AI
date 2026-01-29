# answer_comparator.py - Llama judge for comparing answers from multiple papers
from typing import Dict, Any, List
from config import GENERATIVE_QA_MODEL, JUDGE_MAX_LENGTH
from rag_pipeline import get_generative_qa

def llama_judge(
    question: str,
    answer1: Dict[str, Any],
    answer2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Use Llama as a judge to compare two answers and select the better one.
    
    Args:
        question: The original user question
        answer1: Response dict from paper 1 with keys: response, paper_id, paper_title
        answer2: Response dict from paper 2 with keys: response, paper_id, paper_title
    
    Returns:
        {
            "winner": "paper1" or "paper2",
            "winner_title": str,
            "reasoning": str (Llama's explanation),
            "comparison": str (full comparison analysis)
        }
    """
    gen_qa = get_generative_qa()
    
    paper1_title = answer1.get("paper_title", answer1.get("paper_id", "Paper 1"))
    paper2_title = answer2.get("paper_title", answer2.get("paper_id", "Paper 2"))
    
    response1 = answer1.get("response", "")
    response2 = answer2.get("response", "")
    
    if "llama" in GENERATIVE_QA_MODEL.lower():
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert research paper analyst. Your task is to compare two answers from different research papers and determine which answer is better for the given question. Consider:
1. Completeness and comprehensiveness
2. Accuracy and specificity
3. Relevance to the question
4. Clarity of explanation
5. Use of concrete details, metrics, or examples

Provide your decision and detailed reasoning.<|eot_id|><|start_header_id|>user<|end_header_id|>

Question: {question}

ANSWER FROM {paper1_title}:
{response1}

ANSWER FROM {paper2_title}:
{response2}

Compare these two answers and determine which is better. Start your response with "WINNER: [Paper Title]" then explain your reasoning in detail.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        result = gen_qa(
            prompt,
            max_new_tokens=JUDGE_MAX_LENGTH,
            do_sample=True,
            temperature=0.6,  # Lower temperature for more focused judgment
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False
        )
        comparison = result[0]["generated_text"].strip()
    else:
        # T5/FLAN fallback (simpler comparison)
        prompt = f"""Compare these answers to "{question}":
Answer 1 ({paper1_title}): {response1[:500]}
Answer 2 ({paper2_title}): {response2[:500]}
Which is better and why?"""
        
        result = gen_qa(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7
        )
        comparison = result[0]["generated_text"].strip()
    
    # Parse winner from comparison
    winner = None
    reasoning = comparison
    
    # Try to extract winner from response
    comparison_lower = comparison.lower()
    
    if "winner:" in comparison_lower:
        # Extract line with WINNER:
        for line in comparison.split('\n'):
            if 'winner:' in line.lower():
                if paper1_title.lower() in line.lower():
                    winner = answer1.get("paper_id", "paper1")
                elif paper2_title.lower() in line.lower():
                    winner = answer2.get("paper_id", "paper2")
                break
    
    # Fallback: count mentions of each paper in positive context
    if not winner:
        paper1_mentions = comparison_lower.count(paper1_title.lower())
        paper2_mentions = comparison_lower.count(paper2_title.lower())
        
        # Simple heuristic: paper mentioned more in positive context
        positive_words = ["better", "comprehensive", "detailed", "superior", "more accurate"]
        paper1_score = sum(1 for word in positive_words if word in comparison_lower and paper1_title.lower() in comparison_lower)
        paper2_score = sum(1 for word in positive_words if word in comparison_lower and paper2_title.lower() in comparison_lower)
        
        if paper1_score > paper2_score:
            winner = answer1.get("paper_id", "paper1")
        elif paper2_score > paper1_score:
            winner = answer2.get("paper_id", "paper2")
        else:
            # Default to paper with longer answer (more comprehensive)
            winner = answer1.get("paper_id", "paper1") if len(response1) >= len(response2) else answer2.get("paper_id", "paper2")
    
    winner_title = paper1_title if winner == answer1.get("paper_id", "paper1") else paper2_title
    
    return {
        "winner": winner,
        "winner_title": winner_title,
        "reasoning": reasoning,
        "comparison": comparison
    }

def score_answer_quality(answer: Dict[str, Any], question: str) -> float:
    """
    Score answer quality based on multiple factors.
    Used for hybrid comparison methods.
    
    Returns: Quality score (0-1)
    """
    response = answer.get("response", "")
    contexts = answer.get("context", {}).get("texts", [])
    
    score = 0.0
    
    # Length score (comprehensive but not too verbose)
    word_count = len(response.split())
    if 30 <= word_count <= 200:
        score += 0.3
    elif word_count > 10:
        score += 0.15
    
    # Context coverage score
    if contexts:
        score += 0.2 * min(len(contexts) / 4.0, 1.0)
    
    # Question term coverage
    question_terms = set(question.lower().split())
    response_terms = set(response.lower().split())
    overlap = len(question_terms & response_terms) / max(len(question_terms), 1)
    score += 0.3 * overlap
    
    # Non-generic response (not just "no context" or error)
    if "no relevant" not in response.lower() and "no content found" not in response.lower():
        score += 0.2
    
    return min(score, 1.0)

def compare_answers(
    question: str,
    answers: List[Dict[str, Any]],
    method: str = "llama_judge"
) -> Dict[str, Any]:
    """
    Compare multiple answers and select the best one.
    
    Args:
        question: Original question
        answers: List of answer dicts (currently supports 2 papers)
        method: "llama_judge", "similarity", or "hybrid"
    
    Returns:
        Comparison result with winner and reasoning
    """
    if len(answers) != 2:
        raise ValueError("Currently only supports comparison of 2 papers")
    
    if method == "llama_judge":
        return llama_judge(question, answers[0], answers[1])
    
    elif method == "similarity":
        # Simple scoring-based comparison
        score1 = score_answer_quality(answers[0], question)
        score2 = score_answer_quality(answers[1], question)
        
        winner = answers[0].get("paper_id", "paper1") if score1 >= score2 else answers[1].get("paper_id", "paper2")
        winner_title = answers[0].get("paper_title", "Paper 1") if score1 >= score2 else answers[1].get("paper_title", "Paper 2")
        
        return {
            "winner": winner,
            "winner_title": winner_title,
            "reasoning": f"Quality scores: {answers[0].get('paper_title', 'Paper 1')}={score1:.2f}, {answers[1].get('paper_title', 'Paper 2')}={score2:.2f}",
            "comparison": f"Selected {winner_title} based on quality scoring (comprehensiveness, relevance, specificity)"
        }
    
    else:  # hybrid
        # Combine Llama judgment with quality scores
        llama_result = llama_judge(question, answers[0], answers[1])
        score1 = score_answer_quality(answers[0], question)
        score2 = score_answer_quality(answers[1], question)
        
        llama_result["quality_scores"] = {
            answers[0].get("paper_id", "paper1"): score1,
            answers[1].get("paper_id", "paper2"): score2
        }
        
        return llama_result
