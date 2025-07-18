"""
Reasoning Datasets for Diffusion LLM

This module implements authentic reasoning task datasets following exact specifications 
from "Latent Diffusion with LLMs for Reasoning" with ZERO mocks or simplifications.

Key Features:
- Arithmetic reasoning: single-digit addition chains (3-5 numbers)
- Spatial reasoning: direction and rotation tasks
- Step-by-step solution generation
- Paper-compliant difficulty levels
- Authentic problem complexity
"""

import torch
from torch.utils.data import Dataset
import random
from typing import Dict, List, Tuple, Optional, Any
import re
import json
import os
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from diffusion_llm.models import BARTTokenizerWrapper
except ImportError:
    # Fallback for environments without full diffusion_llm module
    BARTTokenizerWrapper = None


class ArithmeticReasoningDataset(Dataset):
    """
    Arithmetic reasoning dataset following exact paper specifications.
    
    Generates single-digit addition chains with 3-5 numbers as specified in the paper.
    Each problem includes step-by-step reasoning chain showing intermediate calculations.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        min_numbers: int = 3,           # Paper specification: 3-5 numbers
        max_numbers: int = 5,           # Paper specification: 3-5 numbers  
        min_digit: int = 1,             # Paper specification: single digits
        max_digit: int = 9,             # Paper specification: single digits
        tokenizer_name: str = "facebook/bart-base",
        max_seq_len: int = 128,
        seed: Optional[int] = None
    ):
        """
        Initialize arithmetic reasoning dataset.
        
        Args:
            num_samples: Number of samples to generate
            min_numbers: Minimum numbers in addition chain (paper: 3)
            max_numbers: Maximum numbers in addition chain (paper: 5)
            min_digit: Minimum digit value (paper: 1)
            max_digit: Maximum digit value (paper: 9)
            tokenizer_name: Tokenizer model name
            max_seq_len: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.min_numbers = min_numbers
        self.max_numbers = max_numbers
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.max_seq_len = max_seq_len
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize tokenizer
        if BARTTokenizerWrapper is not None:
            self.tokenizer = BARTTokenizerWrapper(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Generate dataset
        self.samples = self._generate_samples()
        
        print(f"📊 ArithmeticReasoningDataset created:")
        print(f"   Samples: {len(self.samples)}")
        print(f"   Number range: {min_numbers}-{max_numbers} numbers")
        print(f"   Digit range: {min_digit}-{max_digit}")
        print(f"   Average solution length: {self._compute_avg_solution_length():.1f} tokens")
    
    def _generate_samples(self) -> List[Dict[str, str]]:
        """Generate arithmetic reasoning samples."""
        samples = []
        
        for _ in range(self.num_samples):
            # Generate problem following paper specifications
            num_count = random.randint(self.min_numbers, self.max_numbers)
            numbers = [random.randint(self.min_digit, self.max_digit) for _ in range(num_count)]
            
            # Create problem string
            problem = " + ".join(map(str, numbers))
            
            # Generate step-by-step solution (exact paper format)
            solution = self._generate_step_by_step_solution(numbers)
            
            samples.append({
                'problem': problem,
                'solution': solution,
                'numbers': numbers,
                'final_answer': sum(numbers)
            })
        
        return samples
    
    def _generate_step_by_step_solution(self, numbers: List[int]) -> str:
        """
        Generate step-by-step solution following exact paper format.
        
        Args:
            numbers: List of numbers to add
            
        Returns:
            Step-by-step reasoning chain
        """
        reasoning_steps = []
        current_sum = numbers[0]
        
        # Start with first number
        reasoning_steps.append(f"Start with {current_sum}")
        
        # Add each subsequent number with explicit step
        for i, num in enumerate(numbers[1:], 1):
            new_sum = current_sum + num
            reasoning_steps.append(f"Step {i}: {current_sum} + {num} = {new_sum}")
            current_sum = new_sum
        
        # Final answer
        reasoning_steps.append(f"Final answer: {current_sum}")
        
        return " → ".join(reasoning_steps)
    
    def _compute_avg_solution_length(self) -> float:
        """Compute average solution length in tokens."""
        total_tokens = 0
        for sample in self.samples:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(sample['solution'])
                total_tokens += len(tokens)
            else:
                # Rough estimate if tokenizer encoding fails
                total_tokens += len(sample['solution'].split())
        
        return total_tokens / len(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize problem and solution
        if hasattr(self.tokenizer, 'encode_reasoning_problem'):
            problem_tokens = self.tokenizer.encode_reasoning_problem(sample['problem'])
            solution_tokens = self.tokenizer.encode_reasoning_solution(sample['solution'])
        else:
            # Fallback tokenization
            problem_tokens = self.tokenizer(
                sample['problem'], 
                truncation=True, 
                padding='max_length',
                max_length=self.max_seq_len,
                return_tensors='pt'
            )
            solution_tokens = self.tokenizer(
                sample['solution'],
                truncation=True,
                padding='max_length', 
                max_length=self.max_seq_len,
                return_tensors='pt'
            )
        
        return {
            'problem_ids': problem_tokens['input_ids'].squeeze(0),
            'problem_mask': problem_tokens['attention_mask'].squeeze(0),
            'solution_ids': solution_tokens['input_ids'].squeeze(0),
            'solution_mask': solution_tokens['attention_mask'].squeeze(0),
            'labels': solution_tokens['input_ids'].squeeze(0),
            'final_answer': torch.tensor(sample['final_answer'], dtype=torch.long),
            'task_type': 'arithmetic'
        }


class SpatialReasoningDataset(Dataset):
    """
    Spatial reasoning dataset following exact paper specifications.
    
    Generates direction and rotation tasks with step-by-step reasoning chains.
    Problems involve starting direction, rotations, and reversals.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        min_operations: int = 2,        # Minimum number of operations
        max_operations: int = 4,        # Maximum number of operations
        tokenizer_name: str = "facebook/bart-base",
        max_seq_len: int = 128,
        seed: Optional[int] = None
    ):
        """
        Initialize spatial reasoning dataset.
        
        Args:
            num_samples: Number of samples to generate
            min_operations: Minimum operations per problem
            max_operations: Maximum operations per problem
            tokenizer_name: Tokenizer model name
            max_seq_len: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.min_operations = min_operations
        self.max_operations = max_operations
        self.max_seq_len = max_seq_len
        
        # Direction mappings (paper standard)
        self.directions = ['up', 'right', 'down', 'left']
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Initialize tokenizer
        if BARTTokenizerWrapper is not None:
            self.tokenizer = BARTTokenizerWrapper(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Generate dataset
        self.samples = self._generate_samples()
        
        print(f"🧭 SpatialReasoningDataset created:")
        print(f"   Samples: {len(self.samples)}")
        print(f"   Operations range: {min_operations}-{max_operations}")
        print(f"   Directions: {self.directions}")
        print(f"   Average solution length: {self._compute_avg_solution_length():.1f} tokens")
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate spatial reasoning samples."""
        samples = []
        
        for _ in range(self.num_samples):
            # Generate problem following paper specifications
            start_direction = random.choice(self.directions)
            num_operations = random.randint(self.min_operations, self.max_operations)
            
            # Generate sequence of operations
            operations = []
            for _ in range(num_operations):
                if random.random() < 0.6:  # 60% rotation, 40% reversal
                    rotation_amount = random.randint(1, 3)
                    clockwise = random.choice([True, False])
                    if clockwise:
                        operations.append(f"rotate {rotation_amount} clockwise")
                    else:
                        operations.append(f"rotate {rotation_amount} counterclockwise")
                else:
                    operations.append("reverse direction")
            
            # Create problem string
            problem = f"Start facing {start_direction}. " + ", then ".join(operations) + "."
            
            # Generate step-by-step solution
            solution, final_direction = self._generate_step_by_step_solution(
                start_direction, operations
            )
            
            samples.append({
                'problem': problem,
                'solution': solution,
                'start_direction': start_direction,
                'operations': operations,
                'final_direction': final_direction
            })
        
        return samples
    
    def _generate_step_by_step_solution(
        self, 
        start_direction: str, 
        operations: List[str]
    ) -> Tuple[str, str]:
        """
        Generate step-by-step spatial reasoning solution.
        
        Args:
            start_direction: Starting direction
            operations: List of operations to perform
            
        Returns:
            Tuple of (solution_string, final_direction)
        """
        reasoning_steps = [f"Start facing {start_direction}"]
        
        current_dir = start_direction
        dir_index = self.directions.index(current_dir)
        
        for operation in operations:
            if 'rotate' in operation:
                # Parse rotation
                if 'clockwise' in operation:
                    rotation_match = re.search(r'rotate (\d+) clockwise', operation)
                    if rotation_match:
                        amount = int(rotation_match.group(1))
                        dir_index = (dir_index + amount) % 4
                        current_dir = self.directions[dir_index]
                        reasoning_steps.append(f"After rotate {amount} clockwise: {current_dir}")
                
                elif 'counterclockwise' in operation:
                    rotation_match = re.search(r'rotate (\d+) counterclockwise', operation)
                    if rotation_match:
                        amount = int(rotation_match.group(1))
                        dir_index = (dir_index - amount) % 4
                        current_dir = self.directions[dir_index]
                        reasoning_steps.append(f"After rotate {amount} counterclockwise: {current_dir}")
            
            elif 'reverse' in operation:
                # Reverse direction (180 degree turn)
                dir_index = (dir_index + 2) % 4
                current_dir = self.directions[dir_index]
                reasoning_steps.append(f"After reverse: {current_dir}")
        
        # Final direction
        reasoning_steps.append(f"Final direction: {current_dir}")
        
        solution = " → ".join(reasoning_steps)
        return solution, current_dir
    
    def _compute_avg_solution_length(self) -> float:
        """Compute average solution length in tokens."""
        total_tokens = 0
        for sample in self.samples:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(sample['solution'])
                total_tokens += len(tokens)
            else:
                total_tokens += len(sample['solution'].split())
        
        return total_tokens / len(self.samples)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize problem and solution
        if hasattr(self.tokenizer, 'encode_reasoning_problem'):
            problem_tokens = self.tokenizer.encode_reasoning_problem(sample['problem'])
            solution_tokens = self.tokenizer.encode_reasoning_solution(sample['solution'])
        else:
            # Fallback tokenization
            problem_tokens = self.tokenizer(
                sample['problem'], 
                truncation=True, 
                padding='max_length',
                max_length=self.max_seq_len,
                return_tensors='pt'
            )
            solution_tokens = self.tokenizer(
                sample['solution'],
                truncation=True,
                padding='max_length', 
                max_length=self.max_seq_len,
                return_tensors='pt'
            )
        
        # Map final direction to index
        final_dir_idx = self.directions.index(sample['final_direction'])
        
        return {
            'problem_ids': problem_tokens['input_ids'].squeeze(0),
            'problem_mask': problem_tokens['attention_mask'].squeeze(0),
            'solution_ids': solution_tokens['input_ids'].squeeze(0),
            'solution_mask': solution_tokens['attention_mask'].squeeze(0),
            'labels': solution_tokens['input_ids'].squeeze(0),
            'final_direction_idx': torch.tensor(final_dir_idx, dtype=torch.long),
            'task_type': 'spatial'
        }


class CombinedReasoningDataset(Dataset):
    """
    Combined dataset with both arithmetic and spatial reasoning tasks.
    
    Balances samples from both task types and provides unified interface
    with task type labels for multi-task training.
    """
    
    def __init__(
        self,
        arithmetic_samples: int = 5000,
        spatial_samples: int = 5000,
        tokenizer_name: str = "facebook/bart-base",
        max_seq_len: int = 128,
        seed: Optional[int] = None
    ):
        """
        Initialize combined reasoning dataset.
        
        Args:
            arithmetic_samples: Number of arithmetic samples
            spatial_samples: Number of spatial samples
            tokenizer_name: Tokenizer model name
            max_seq_len: Maximum sequence length
            seed: Random seed for reproducibility
        """
        self.arithmetic_samples = arithmetic_samples
        self.spatial_samples = spatial_samples
        self.max_seq_len = max_seq_len
        
        # Task type mapping
        self.task_types = ['arithmetic', 'spatial']
        self.task_type_to_id = {task: i for i, task in enumerate(self.task_types)}
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        # Create individual datasets
        self.arithmetic_dataset = ArithmeticReasoningDataset(
            num_samples=arithmetic_samples,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            seed=seed
        )
        
        self.spatial_dataset = SpatialReasoningDataset(
            num_samples=spatial_samples,
            tokenizer_name=tokenizer_name,
            max_seq_len=max_seq_len,
            seed=seed
        )
        
        # Create combined index mapping
        self.sample_mapping = self._create_sample_mapping()
        
        print(f"🔄 CombinedReasoningDataset created:")
        print(f"   Arithmetic samples: {arithmetic_samples}")
        print(f"   Spatial samples: {spatial_samples}")
        print(f"   Total samples: {len(self.sample_mapping)}")
        print(f"   Task types: {self.task_types}")
    
    def _create_sample_mapping(self) -> List[Tuple[str, int]]:
        """Create mapping from combined index to dataset and sample index."""
        mapping = []
        
        # Add arithmetic samples
        for i in range(len(self.arithmetic_dataset)):
            mapping.append(('arithmetic', i))
        
        # Add spatial samples
        for i in range(len(self.spatial_dataset)):
            mapping.append(('spatial', i))
        
        # Shuffle for balanced training
        random.shuffle(mapping)
        
        return mapping
    
    def __len__(self) -> int:
        return len(self.sample_mapping)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        task_type, sample_idx = self.sample_mapping[idx]
        
        if task_type == 'arithmetic':
            sample = self.arithmetic_dataset[sample_idx]
        elif task_type == 'spatial':
            sample = self.spatial_dataset[sample_idx]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Add task type ID
        sample['task_type_id'] = torch.tensor(
            self.task_type_to_id[task_type], 
            dtype=torch.long
        )
        sample['task_type'] = task_type
        
        return sample
    
    def get_task_distribution(self) -> Dict[str, int]:
        """Get distribution of task types in dataset."""
        distribution = {}
        for task_type, _ in self.sample_mapping:
            distribution[task_type] = distribution.get(task_type, 0) + 1
        return distribution 