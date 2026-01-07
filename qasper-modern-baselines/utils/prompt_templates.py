"""
Prompt templates for different question answering strategies
"""

class PromptTemplates:
    """Collection of prompt templates for Qasper QA"""
    
    @staticmethod
    def get_basic_prompt(title: str, full_text: str, question: str) -> str:
        """
        Basic prompt template for answering questions
        
        Args:
            title: Paper title
            full_text: Full paper text
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert at answering questions about scientific research papers.

Paper Title: {title}

Paper Content:
{full_text}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information in the paper.
2. If the paper doesn't contain enough information to answer, respond with "UNANSWERABLE".
3. For yes/no questions, respond with ONLY "Yes" or "No" - do not add explanations.
4. For other questions, be concise and specific in your answer.

Answer:"""
    
    @staticmethod
    def get_evidence_prompt(title: str, full_text: str, question: str) -> str:
        """
        Prompt template that asks for evidence selection
        
        Args:
            title: Paper title
            full_text: Full paper text
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert at answering questions about scientific research papers.

Paper Title: {title}

Paper Content:
{full_text}

Question: {question}

Instructions:
1. First, identify the relevant paragraph(s) that contain evidence for answering the question.
2. Then, provide your answer based on that evidence.
3. Format your response as:

EVIDENCE: [quote the most relevant paragraph]

ANSWER: [your answer based on the evidence]

4. If the paper doesn't contain enough information, respond with "UNANSWERABLE".

Response:"""
    
    @staticmethod
    def get_cot_prompt(title: str, full_text: str, question: str) -> str:
        """
        Chain-of-thought prompt for complex reasoning
        
        Args:
            title: Paper title
            full_text: Full paper text
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert at answering questions about scientific research papers.

Paper Title: {title}

Paper Content:
{full_text}

Question: {question}

Instructions:
1. Think step-by-step about what information is needed to answer the question.
2. Find the relevant information in the paper.
3. Reason through the answer based on the evidence.
4. Provide your final answer.

Let's think through this step by step:

Step 1 - Understanding the question:
Step 2 - Finding relevant information:
Step 3 - Reasoning:
Step 4 - Final Answer:"""
    
    @staticmethod
    def get_context_window_prompt(title: str, abstract: str, relevant_sections: str, question: str) -> str:
        """
        Prompt for models with limited context windows
        Uses abstract + relevant sections only
        
        Args:
            title: Paper title
            abstract: Paper abstract
            relevant_sections: Pre-selected relevant sections
            question: Question to answer
            
        Returns:
            Formatted prompt string
        """
        return f"""You are an expert at answering questions about scientific research papers.

Paper Title: {title}

Abstract:
{abstract}

Relevant Sections:
{relevant_sections}

Question: {question}

Instructions:
Answer the question based on the provided abstract and relevant sections. If the information is not sufficient, respond with "UNANSWERABLE".

Answer:"""
    
    @staticmethod
    def clean_yes_no_answer(answer: str, question_type: str) -> str:
        """
        Clean up Yes/No answers by removing verbose explanations
        
        Args:
            answer: Raw model output
            question_type: Type of question (e.g., 'yes_no')
            
        Returns:
            Cleaned answer (just "Yes", "No", or "Unanswerable" for yes/no questions)
        """
        if question_type != 'yes_no':
            return answer
        
        # Clean the answer
        answer_lower = answer.lower().strip()
        
        # Extract just the yes/no part
        # Check first sentence or first few words
        first_sentence = answer_lower.split('.')[0].split('\n')[0]
        first_words = first_sentence.split()[:10]
        
        # Look for yes/no indicators
        yes_indicators = ['yes', 'true', 'correct', 'affirmative', 'indeed', 'certainly']
        no_indicators = ['no', 'false', 'incorrect', 'negative', 'not', "doesn't", "don't"]
        
        # Count indicators in first part of answer
        yes_count = sum(1 for word in first_words if any(ind in word for ind in yes_indicators))
        no_count = sum(1 for word in first_words if any(ind in word for ind in no_indicators))
        
        # Check for unanswerable
        if any(word in answer_lower for word in ['unanswerable', 'cannot answer', 'insufficient', 'unclear']):
            return "Unanswerable"
        
        # Determine answer based on indicators
        if yes_count > no_count:
            return "Yes"
        elif no_count > yes_count:
            return "No"
        elif answer.strip().lower().startswith('yes'):
            return "Yes"
        elif answer.strip().lower().startswith('no'):
            return "No"
        
        # If still unclear, return original (better to keep full answer than guess wrong)
        return answer
