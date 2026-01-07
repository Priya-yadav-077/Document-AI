"""
Data loader for Qasper dataset
Handles loading and preprocessing of Qasper JSON files
"""

import json
import random
from typing import List, Dict, Optional
from pathlib import Path


class QasperDataLoader:
    """Load and preprocess Qasper dataset"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to Qasper JSON file
        """
        self.data_path = Path(data_path)
        self.data = None
    
    def load(self, max_samples: Optional[int] = None, random_seed: Optional[int] = None) -> List[Dict]:
        """
        Load Qasper data from JSON file
        
        Args:
            max_samples: Maximum number of papers to load (None = all)
            random_seed: Random seed for sampling (None = sequential, int = random with seed)
            
        Returns:
            List of paper dictionaries with questions and answers
        """
        with open(self.data_path, 'r') as f:
            raw_data = json.load(f)
        
        papers = []
        
        # Get all paper IDs
        all_paper_ids = list(raw_data.keys())
        
        # If max_samples specified and random_seed provided, randomly sample
        if max_samples and random_seed is not None:
            if max_samples < len(all_paper_ids):
                random.seed(random_seed)
                selected_paper_ids = random.sample(all_paper_ids, max_samples)
                print(f"📊 Randomly sampled {max_samples} papers (seed={random_seed})")
            else:
                selected_paper_ids = all_paper_ids
        elif max_samples:
            # Sequential sampling (first N papers)
            selected_paper_ids = all_paper_ids[:max_samples]
            print(f"📊 Using first {max_samples} papers (sequential sampling)")
        else:
            # All papers
            selected_paper_ids = all_paper_ids
        
        for paper_id in selected_paper_ids:
            paper_data = raw_data[paper_id]
            
            # Extract full text
            full_text = self._extract_full_text(paper_data)
            
            # Extract questions and answers
            questions = self._extract_questions(paper_data, paper_id)
            
            paper_dict = {
                'paper_id': paper_id,
                'title': paper_data.get('title', ''),
                'abstract': paper_data.get('abstract', ''),
                'full_text': full_text,
                'full_text_sections': paper_data.get('full_text', {}),
                'questions': questions
            }
            
            papers.append(paper_dict)
        
        self.data = papers
        return papers
    
    def _extract_full_text(self, paper_data: Dict) -> str:
        """
        Extract and concatenate full text from paper
        
        Args:
            paper_data: Raw paper dictionary
            
        Returns:
            Full text as single string
        """
        full_text = paper_data.get('abstract', '') + "\n\n"
        
        # Handle full_text which is a dict with 'section_name' and 'paragraphs'
        full_text_data = paper_data.get('full_text', {})
        
        if isinstance(full_text_data, dict):
            # New format: {section_name: [...], paragraphs: [[...], ...]}
            section_names = full_text_data.get('section_name', [])
            all_paragraphs = full_text_data.get('paragraphs', [])
            
            # Zip sections and paragraphs together
            for i, paragraphs in enumerate(all_paragraphs):
                if i < len(section_names) and section_names[i]:
                    full_text += f"\n{section_names[i]}\n"
                
                for para in paragraphs:
                    full_text += para + "\n"
        elif isinstance(full_text_data, list):
            # Old format: [{section_name: ..., paragraphs: [...]}, ...]
            for section in full_text_data:
                section_name = section.get('section_name', '')
                paragraphs = section.get('paragraphs', [])
                
                if section_name:
                    full_text += f"\n{section_name}\n"
                
                for para in paragraphs:
                    full_text += para + "\n"
        
        return full_text.strip()
    
    def _extract_questions(self, paper_data: Dict, paper_id: str) -> List[Dict]:
        """
        Extract questions and answers for a paper
        
        Args:
            paper_data: Raw paper dictionary
            paper_id: Paper identifier
            
        Returns:
            List of question dictionaries
        """
        questions = []
        
        qas_data = paper_data.get('qas', {})
        
        # Handle new format: {question: [...], answers: [...], ...}
        if isinstance(qas_data, dict) and 'question' in qas_data:
            question_list = qas_data.get('question', [])
            question_id_list = qas_data.get('question_id', [])
            answers_list = qas_data.get('answers', [])
            
            for idx in range(len(question_list)):
                question_text = question_list[idx] if idx < len(question_list) else ''
                question_id = question_id_list[idx] if idx < len(question_id_list) else f"{paper_id}_q{idx}"
                answers_data = answers_list[idx] if idx < len(answers_list) else {'answer': []}
                
                # Extract answer annotations
                answer_annotations = answers_data.get('answer', [])
                
                # Determine question type
                question_type = self._determine_question_type(answer_annotations)
                
                # Extract gold answers
                gold_answers = []
                gold_evidence = []
                
                for ans in answer_annotations:
                    # Check if unanswerable
                    if ans.get('unanswerable', False):
                        gold_answers.append('UNANSWERABLE')
                        gold_evidence.append([])
                        continue
                    
                    # Extract answer text
                    if ans.get('yes_no') is not None:
                        ans_text = 'Yes' if ans['yes_no'] else 'No'
                    elif ans.get('extractive_spans'):
                        ans_text = ' '.join(ans['extractive_spans'])
                    elif ans.get('free_form_answer'):
                        ans_text = ans['free_form_answer']
                    else:
                        ans_text = ''
                    
                    gold_answers.append(ans_text)
                    
                    # Extract evidence
                    evidence = ans.get('evidence', [])
                    gold_evidence.append(evidence)
                
                question_dict = {
                    'question_id': question_id,
                    'paper_id': paper_id,
                    'question': question_text,
                    'question_type': question_type,
                    'gold_answers': gold_answers,
                    'gold_evidence': gold_evidence
                }
                
                questions.append(question_dict)
        
        # Handle old format (if any): [question: {...}, answers: {...}, ...]
        elif isinstance(qas_data, list):
            for qa_idx, qa in enumerate(qas_data):
                question_text = qa.get('question', '')
                question_id = qa.get('question_id', f"{paper_id}_q{qa_idx}")
                answers_data = qa.get('answers', [])
                
                # Determine question type
                question_type = self._determine_question_type(answers_data)
                
                # Extract gold answers and evidence (same logic as above)
                gold_answers = []
                gold_evidence = []
                
                for ans in answers_data:
                    if ans.get('unanswerable', False):
                        gold_answers.append('UNANSWERABLE')
                        gold_evidence.append([])
                        continue
                    
                    if ans.get('yes_no') is not None:
                        ans_text = 'Yes' if ans['yes_no'] else 'No'
                    elif ans.get('extractive_spans'):
                        ans_text = ' '.join(ans['extractive_spans'])
                    elif ans.get('free_form_answer'):
                        ans_text = ans['free_form_answer']
                    else:
                        ans_text = ''
                    
                    gold_answers.append(ans_text)
                    evidence = ans.get('evidence', [])
                    gold_evidence.append(evidence)
                
                question_dict = {
                    'question_id': question_id,
                    'paper_id': paper_id,
                    'question': question_text,
                    'question_type': question_type,
                    'gold_answers': gold_answers,
                    'gold_evidence': gold_evidence
                }
                
                questions.append(question_dict)
        
        return questions
    
    def _determine_question_type(self, answers_data: List[Dict]) -> str:
        """
        Determine the type of question based on answers
        
        Args:
            answers_data: List of answer annotations
            
        Returns:
            Question type: 'yes_no', 'extractive', 'abstractive', or 'unanswerable'
        """
        if not answers_data:
            return 'unknown'
        
        first_answer = answers_data[0]
        
        if first_answer.get('unanswerable', False):
            return 'unanswerable'
        elif first_answer.get('yes_no') is not None:
            return 'yes_no'
        elif first_answer.get('extractive_spans'):
            return 'extractive'
        elif first_answer.get('free_form_answer'):
            return 'abstractive'
        else:
            return 'unknown'
            return 'unknown'
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded data
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.data:
            return {}
        
        total_papers = len(self.data)
        total_questions = sum(len(paper['questions']) for paper in self.data)
        
        question_types = {}
        for paper in self.data:
            for q in paper['questions']:
                q_type = q['question_type']
                question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return {
            'total_papers': total_papers,
            'total_questions': total_questions,
            'questions_per_paper': total_questions / total_papers if total_papers > 0 else 0,
            'question_types': question_types
        }


def load_qasper_data(data_path: str, max_samples: Optional[int] = None, random_seed: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to load Qasper data
    
    Args:
        data_path: Path to Qasper JSON file
        max_samples: Maximum number of papers to load (None = all)
        random_seed: Random seed for sampling (None = sequential, int = random with seed)
        
    Returns:
        List of paper dictionaries
    """
    loader = QasperDataLoader(data_path)
    papers = loader.load(max_samples=max_samples, random_seed=random_seed)
    
    stats = loader.get_statistics()
    print(f"Loaded {stats['total_papers']} papers with {stats['total_questions']} questions")
    print(f"Question type distribution: {stats['question_types']}")
    
    return papers


if __name__ == "__main__":
    # Test data loader
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        papers = load_qasper_data(data_path, max_samples=5)
        
        # Print sample
        print("\nSample paper:")
        paper = papers[0]
        print(f"Title: {paper['title']}")
        print(f"Number of questions: {len(paper['questions'])}")
        print(f"\nFirst question: {paper['questions'][0]['question']}")
        print(f"Gold answers: {paper['questions'][0]['gold_answers']}")
    else:
        print("Usage: python data_loader.py <path_to_qasper_json>")
