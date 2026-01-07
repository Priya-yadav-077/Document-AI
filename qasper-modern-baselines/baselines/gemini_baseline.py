"""
Gemini baseline implementations for Qasper
Supports both Gemini 1.5 Pro and Gemini 2.0 Flash
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.prompt_templates import PromptTemplates

# Load environment variables from .env file (check current and parent directories)
load_dotenv()
load_dotenv(Path(__file__).parent.parent / '.env')
load_dotenv(Path(__file__).parent.parent.parent / '.env')


class GeminiBaseline:
    """Gemini baseline for Qasper question answering"""
    
    def __init__(
        self,
        model: str = "gemini-1.5-pro",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 1024,
        use_evidence_prompt: bool = False
    ):
        """
        Initialize Gemini baseline
        
        Args:
            model: Model name ("gemini-1.5-pro", "gemini-2.0-flash", etc.)
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            temperature: Sampling temperature (0.0 = deterministic)
            max_output_tokens: Maximum tokens in response
            use_evidence_prompt: Whether to use evidence-based prompting
        """
        self.model_name = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.use_evidence_prompt = use_evidence_prompt
        
        if not self.api_key:
            raise ValueError("Google API key not provided. Set GOOGLE_API_KEY environment variable.")
        
        # Map model names to correct API format
        # Your API has access to these models (tested Dec 2025)
        model_map = {
            "gemini-1.5-pro": "gemini-pro-latest",
            "gemini-1.5-flash": "gemini-flash-latest",
            "gemini-2.0-flash": "gemini-2.0-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-pro": "gemini-pro-latest"
        }
        
        # Use mapped name if available, otherwise use as-is
        api_model_name = model_map.get(model, model)
        
        # Initialize model
        self.model = ChatGoogleGenerativeAI(
            model=api_model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens
        )
        
        self.prompt_template = PromptTemplates()
    
    def answer_question(
        self,
        title: str,
        full_text: str,
        question: str,
        max_context_length: Optional[int] = None
    ) -> Dict:
        """
        Answer a single question about a paper
        
        Args:
            title: Paper title
            full_text: Full paper text
            question: Question to answer
            max_context_length: Maximum context length (chars), truncate if needed
            
        Returns:
            Dictionary with answer and metadata
        """
        # Truncate if needed
        if max_context_length and len(full_text) > max_context_length:
            full_text = full_text[:max_context_length] + "\n[...truncated...]"
        
        # Build prompt
        if self.use_evidence_prompt:
            prompt = self.prompt_template.get_evidence_prompt(title, full_text, question)
        else:
            prompt = self.prompt_template.get_basic_prompt(title, full_text, question)
        
        # Get response
        try:
            start_time = time.time()
            response = self.model.invoke(prompt)
            latency = time.time() - start_time
            
            answer_text = response.content
            
            # Parse evidence if using evidence prompt
            evidence = None
            if self.use_evidence_prompt:
                # Try to parse EVIDENCE: ... ANSWER: ... format
                if "EVIDENCE:" in answer_text and "ANSWER:" in answer_text:
                    parts = answer_text.split("ANSWER:")
                    if len(parts) >= 2:
                        evidence_text = parts[0].replace("EVIDENCE:", "").strip()
                        answer_text = parts[1].strip()
                        
                        # Split evidence into paragraphs (if multiple provided)
                        # Models may provide multiple paragraphs separated by newlines
                        if evidence_text:
                            evidence = [p.strip() for p in evidence_text.split("\n\n") if p.strip()]
                            if not evidence:  # Try single newline split
                                evidence = [p.strip() for p in evidence_text.split("\n") if p.strip()]
                elif "EVIDENCE:" in answer_text:
                    # Model provided evidence but no ANSWER: marker
                    evidence_text = answer_text.replace("EVIDENCE:", "").strip()
                    if evidence_text:
                        evidence = [p.strip() for p in evidence_text.split("\n\n") if p.strip()]
                        if not evidence:
                            evidence = [p.strip() for p in evidence_text.split("\n") if p.strip()]
                    answer_text = ""  # No answer provided
            
            # Clean yes/no answers to avoid verbose explanations hurting F1
            # This will be applied later when we have question_type metadata
            
            return {
                'answer': answer_text,
                'evidence': evidence,
                'model': self.model_name,
                'latency': latency,
                'success': True,
                'error': None
            }
        
        except Exception as e:
            error_msg = str(e)
            # Check for quota/rate limit errors
            is_quota_error = any(indicator in error_msg.upper() for indicator in [
                'RESOURCE_EXHAUSTED',
                'QUOTA',
                'RATE_LIMIT',
                '429'
            ])
            
            return {
                'answer': '',
                'evidence': None,
                'model': self.model_name,
                'latency': 0.0,
                'success': False,
                'error': error_msg,
                'is_quota_error': is_quota_error
            }
    
    def answer_batch(
        self,
        papers: List[Dict],
        rate_limit_delay: float = 1.0,
        verbose: bool = True,
        stop_on_quota_error: bool = True
    ) -> List[Dict]:
        """
        Answer all questions for a batch of papers
        
        Args:
            papers: List of paper dictionaries (from data_loader)
            rate_limit_delay: Delay between API calls (seconds)
            verbose: Whether to print progress
            stop_on_quota_error: Whether to stop when quota is exhausted
            
        Returns:
            List of results dictionaries
        """
        results = []
        total_questions = sum(len(paper['questions']) for paper in papers)
        question_count = 0
        consecutive_quota_errors = 0
        
        for paper in papers:
            paper_id = paper['paper_id']
            title = paper['title']
            full_text = paper['full_text']
            
            for question_data in paper['questions']:
                question_count += 1
                
                if verbose:
                    print(f"[{question_count}/{total_questions}] Processing {question_data['question_id']}...")
                
                # Answer question
                result = self.answer_question(title, full_text, question_data['question'])
                
                # Clean yes/no answers to remove verbose explanations
                if question_data['question_type'] == 'yes_no' and result.get('success'):
                    result['answer'] = self.prompt_template.clean_yes_no_answer(
                        result['answer'], 
                        question_data['question_type']
                    )
                
                # Add question metadata
                result.update({
                    'question_id': question_data['question_id'],
                    'paper_id': paper_id,
                    'question': question_data['question'],
                    'question_type': question_data['question_type'],
                    'gold_answers': question_data['gold_answers'],
                    'gold_evidence': question_data['gold_evidence']
                })
                
                results.append(result)
                
                # Check for quota errors
                if result.get('is_quota_error', False):
                    consecutive_quota_errors += 1
                    if verbose:
                        print(f"⚠️  Quota/rate limit error detected ({consecutive_quota_errors} consecutive)")
                        print(f"   Error: {result.get('error', 'Unknown')}")
                    
                    # Stop immediately on first clear quota error, or after 2 transient errors
                    should_stop = False
                    error_msg = result.get('error', '').upper()
                    
                    # Immediate stop conditions (clear quota exhaustion)
                    if any(indicator in error_msg for indicator in [
                        'RESOURCE_EXHAUSTED',
                        'QUOTA_EXCEEDED', 
                        'INSUFFICIENT_QUOTA'
                    ]):
                        should_stop = True
                        if verbose:
                            print(f"   ⚠️  Clear quota exhaustion detected - stopping immediately")
                    # Stop after 2 consecutive errors (could be rate limiting)
                    elif consecutive_quota_errors >= 2:
                        should_stop = True
                    
                    if stop_on_quota_error and should_stop:
                        if verbose:
                            print(f"\n{'='*80}")
                            print(f"❌ STOPPING: API quota exhausted!")
                            print(f"{'='*80}")
                            print(f"   Processed {question_count}/{total_questions} questions")
                            print(f"   Successful predictions: {len([r for r in results if r.get('success')])}")
                            print(f"   Failed predictions: {len([r for r in results if not r.get('success')])}")
                            print(f"   Results saved with {len(results)} predictions.")
                            print(f"\n💡 TIP: To avoid quotas, use local Hugging Face models:")
                            print(f"   python scripts/run_evaluation.py --model mistralai/Mistral-7B-Instruct-v0.2")
                        break
                else:
                    consecutive_quota_errors = 0  # Reset counter on success
                
                # Rate limiting
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)
            
            # Break outer loop if quota exhausted
            if stop_on_quota_error and consecutive_quota_errors >= 1:
                # Check if we should stop
                if results and results[-1].get('is_quota_error', False):
                    error_msg = results[-1].get('error', '').upper()
                    should_stop = any(indicator in error_msg for indicator in [
                        'RESOURCE_EXHAUSTED',
                        'QUOTA_EXCEEDED',
                        'INSUFFICIENT_QUOTA'
                    ]) or consecutive_quota_errors >= 2
                    
                    if should_stop:
                        break
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the model
        
        Returns:
            Dictionary with model metadata
        """
        context_windows = {
            "gemini-1.5-pro": 2_000_000,
            "gemini-2.0-flash": 1_000_000,
            "gemini-1.5-flash": 1_000_000
        }
        
        return {
            'model_name': self.model_name,
            'provider': 'Google',
            'context_window': context_windows.get(self.model_name, 'unknown'),
            'temperature': self.temperature,
            'max_output_tokens': self.max_output_tokens,
            'use_evidence_prompt': self.use_evidence_prompt
        }


if __name__ == "__main__":
    # Test the baseline
    from utils.data_loader import load_qasper_data
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
        # Load sample data
        papers = load_qasper_data(data_path, max_samples=1)
        
        # Initialize baseline
        baseline = GeminiBaseline(model="gemini-2.0-flash")
        
        # Test on first question
        paper = papers[0]
        question = paper['questions'][0]
        
        print("\n" + "="*80)
        print(f"Testing Gemini Baseline")
        print("="*80)
        print(f"\nPaper: {paper['title']}")
        print(f"Question: {question['question']}")
        print(f"Question Type: {question['question_type']}")
        print(f"Gold Answers: {question['gold_answers']}")
        
        result = baseline.answer_question(
            paper['title'],
            paper['full_text'],
            question['question']
        )
        
        print(f"\nPredicted Answer: {result['answer']}")
        print(f"Latency: {result['latency']:.2f}s")
        print(f"Success: {result['success']}")
        
    else:
        print("Usage: python gemini_baseline.py <path_to_qasper_json>")
