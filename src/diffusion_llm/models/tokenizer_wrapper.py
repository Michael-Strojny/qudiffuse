"""
BART Tokenizer Wrapper for Reasoning Tasks

Provides specialized tokenization for arithmetic and spatial reasoning tasks
with step-by-step solution formatting.

ZERO mocks, ZERO simplifications, ZERO placeholders.
"""

import torch
from transformers import BartTokenizer
from typing import List, Dict, Any, Optional, Tuple
import re
import logging

logger = logging.getLogger(__name__)


class BARTTokenizerWrapper:
    """Enhanced BART tokenizer for reasoning tasks with step-by-step processing."""
    
    def __init__(self, model_name: str = "facebook/bart-base", max_length: int = 256):
        """
        Initialize BART tokenizer wrapper.
        
        Args:
            model_name: BART model name
            max_length: Maximum sequence length
        """
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Add special tokens for reasoning
        self.special_tokens = {
            'step_sep': '<STEP>',
            'answer_start': '<ANS>',
            'problem_start': '<PROB>',
            'reasoning_start': '<REASON>'
        }
        
        # Add special tokens to tokenizer
        self.tokenizer.add_special_tokens({
            'additional_special_tokens': list(self.special_tokens.values())
        })
        
        logger.info(f"Initialized BART tokenizer with {len(self.tokenizer)} tokens")
    
    def __len__(self) -> int:
        """Return the vocabulary size of the underlying tokenizer."""
        return len(self.tokenizer)
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Encode plain text for processing.
        
        Args:
            text: Text to encode
            max_length: Maximum sequence length (uses self.max_length if not provided)
            
        Returns:
            Dictionary with tokenized inputs
        """
        if max_length is None:
            max_length = self.max_length
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
    
    def encode_problem(self, problem: str, problem_type: str = "arithmetic") -> Dict[str, torch.Tensor]:
        """
        Encode a reasoning problem for processing.
        
        Args:
            problem: Problem statement
            problem_type: Type of reasoning problem
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Format problem with special tokens
        formatted_problem = f"{self.special_tokens['problem_start']} {problem}"
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_problem,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'problem_type': problem_type
        }
    
    def encode_reasoning_chain(self, problem: str, steps: List[str], answer: str) -> Dict[str, torch.Tensor]:
        """
        Encode a complete reasoning chain with steps.
        
        Args:
            problem: Original problem
            steps: List of reasoning steps
            answer: Final answer
            
        Returns:
            Dictionary with tokenized reasoning chain
        """
        # Build step-by-step solution
        reasoning_text = f"{self.special_tokens['problem_start']} {problem} "
        reasoning_text += f"{self.special_tokens['reasoning_start']} "
        
        for i, step in enumerate(steps):
            reasoning_text += f"{self.special_tokens['step_sep']} Step {i+1}: {step} "
        
        reasoning_text += f"{self.special_tokens['answer_start']} {answer}"
        
        # Tokenize complete reasoning chain
        encoded = self.tokenizer(
            reasoning_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'reasoning_length': len(steps)
        }
    
    def decode_reasoning_output(self, token_ids: torch.Tensor) -> str:
        """
        Decode generated token IDs back to reasoning text.
        
        Args:
            token_ids: Generated token IDs
            
        Returns:
            Decoded reasoning text
        """
        # Decode tokens
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # Clean up special tokens for readability
        for token_name, token in self.special_tokens.items():
            text = text.replace(token, f" [{token_name.upper()}] ")
        
        return text.strip()
    
    def extract_answer(self, generated_text: str) -> Optional[str]:
        """
        Extract the final answer from generated reasoning text.
        
        Args:
            generated_text: Generated reasoning text
            
        Returns:
            Extracted answer or None
        """
        # Look for answer after answer start token
        answer_pattern = rf"{re.escape(self.special_tokens['answer_start'])}\s*(.+?)(?:\s*$|\s*<|$)"
        match = re.search(answer_pattern, generated_text)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: look for "Answer:" pattern
        fallback_pattern = r"Answer:\s*(.+?)(?:\s*$|\s*<|$)"
        match = re.search(fallback_pattern, generated_text, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return None
    
    def parse_reasoning_steps(self, generated_text: str) -> List[str]:
        """
        Parse individual reasoning steps from generated text.
        
        Args:
            generated_text: Generated reasoning text
            
        Returns:
            List of reasoning steps
        """
        steps = []
        
        # Find all step patterns
        step_pattern = rf"{re.escape(self.special_tokens['step_sep'])}\s*Step\s*\d+:\s*(.+?)(?={re.escape(self.special_tokens['step_sep'])}|{re.escape(self.special_tokens['answer_start'])}|$)"
        matches = re.finditer(step_pattern, generated_text, re.IGNORECASE)
        
        for match in matches:
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)
        
        return steps
    
    def validate_reasoning_format(self, text: str) -> bool:
        """
        Validate that reasoning text follows expected format.
        
        Args:
            text: Text to validate
            
        Returns:
            True if format is valid
        """
        # Check for required components
        has_problem = self.special_tokens['problem_start'] in text
        has_reasoning = self.special_tokens['reasoning_start'] in text
        has_answer = self.special_tokens['answer_start'] in text
        
        return has_problem and has_reasoning and has_answer
    
    def get_vocabulary_size(self) -> int:
        """Get the vocabulary size of the tokenizer."""
        return len(self.tokenizer)
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get token IDs for special tokens."""
        return {
            name: self.tokenizer.convert_tokens_to_ids(token)
            for name, token in self.special_tokens.items()
        } 