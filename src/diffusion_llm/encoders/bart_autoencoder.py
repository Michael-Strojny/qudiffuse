"""
BART Binary AutoEncoder for Diffusion LLM

This module implements a BART-based autoencoder that integrates with our QuDiffuse
binary diffusion system. It follows the paper specification for "Latent Diffusion 
with LLMs for Reasoning" while maintaining compatibility with our quantum annealer backend.

Key Features:
- BART encoder-decoder for text processing
- Perceiver-based compression to fixed binary latents (lae=16, dae=256)
- Integration with QuDiffuse binary diffusion system
- Support for reasoning tasks (arithmetic, spatial)
- Two-stage training compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, Any, Union
from transformers import BartForConditionalGeneration, BartConfig, AutoTokenizer
from .perceiver_ae import PerceiverBinaryAutoEncoder

logger = logging.getLogger(__name__)


class BARTBinaryAutoEncoder(nn.Module):
    """
    BART-based autoencoder with binary latent compression for reasoning tasks.
    
    This class implements the core architecture from "Latent Diffusion with LLMs for Reasoning":
    1. BART encoder processes input text sequences
    2. Perceiver autoencoder compresses to fixed binary latents
    3. Integration with QuDiffuse binary diffusion for reasoning
    4. BART decoder reconstructs reasoning chains
    
    Architecture follows paper specifications:
    - lae = 16 (fixed latent sequence length)
    - dae = 256 (latent dimension)
    - Binary quantization for quantum annealer compatibility
    """
    
    def __init__(
        self,
        bart_model_name: str = "facebook/bart-base",
        num_encoder_latents: int = 16,      # lae from paper
        num_decoder_latents: int = 32,      # decoder latent length
        dim_ae: int = 256,                  # dae from paper
        perceiver_depth: int = 6,           # Perceiver transformer depth
        perceiver_heads: int = 8,           # Number of attention heads
        dropout: float = 0.1,               # Dropout rate
        freeze_bart_encoder: bool = False,  # Freeze BART encoder during Stage 2
        l2_normalize_latents: bool = False, # L2 normalize latents
        binary_quantization: bool = True    # Enable binary quantization
    ):
        super().__init__()
        
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.freeze_bart_encoder = freeze_bart_encoder
        self.binary_quantization = binary_quantization
        
        # Load BART model and configuration
        self.bart_config = BartConfig.from_pretrained(bart_model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Get BART dimensions
        self.dim_lm = self.bart_config.d_model  # Usually 768 for BART-base
        
        # Perceiver autoencoder for binary latent compression
        self.perceiver_ae = PerceiverBinaryAutoEncoder(
            dim_lm=self.dim_lm,
            dim_ae=dim_ae,
            num_encoder_latents=num_encoder_latents,
            num_decoder_latents=num_decoder_latents,
            depth=perceiver_depth,
            num_heads=perceiver_heads,
            dropout=dropout,
            binary_quantization=binary_quantization,
            l2_normalize_latents=l2_normalize_latents
        )
        
        # Resize token embeddings to match tokenizer vocabulary
        # Note: This must be done BEFORE freezing parameters
        original_vocab_size = self.bart_model.config.vocab_size
        logger.info(f"Original BART vocab size: {original_vocab_size}")
        
        # Freeze BART encoder if requested (for Stage 2 training)
        if freeze_bart_encoder:
            for param in self.bart_model.get_encoder().parameters():
                param.requires_grad = False
        
        print(f"🚀 BARTBinaryAutoEncoder initialized:")
        print(f"   BART model: {bart_model_name}")
        print(f"   LM dimension: {self.dim_lm}")
        print(f"   Latent dimension: {dim_ae}")
        print(f"   Encoder latents: {num_encoder_latents}")
        print(f"   BART encoder frozen: {freeze_bart_encoder}")
        print(f"   Binary quantization: {binary_quantization}")
    
    def get_encoder(self):
        """Get BART encoder for external use."""
        return self.bart_model.get_encoder()
    
    def get_decoder(self):
        """Get BART decoder for external use."""
        return self.bart_model.get_decoder()
    
    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """
        Resize token embeddings to match tokenizer vocabulary.
        
        Args:
            new_vocab_size: New vocabulary size from tokenizer
        """
        current_vocab_size = self.bart_model.config.vocab_size
        if new_vocab_size != current_vocab_size:
            logger.info(f"Resizing BART embeddings: {current_vocab_size} → {new_vocab_size}")
            self.bart_model.resize_token_embeddings(new_vocab_size)
            logger.info("✅ BART embeddings resized successfully")
    
    def encode_text_to_binary_latents(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode text to binary latents for diffusion.
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Binary latents [B, num_encoder_latents, dim_ae]
        """
        # BART encoder
        with torch.set_grad_enabled(not self.freeze_bart_encoder):
            encoder_outputs = self.bart_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Compress to binary latents via Perceiver
        binary_latents = self.perceiver_ae.get_binary_latents_for_diffusion(
            encoder_outputs.last_hidden_state,
            attention_mask
        )
        
        return binary_latents
    
    def decode_binary_latents_to_text(
        self, 
        binary_latents: torch.Tensor,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Decode binary latents back to text tokens.
        
        Args:
            binary_latents: Binary latents [B, num_encoder_latents, dim_ae]
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated token IDs [B, output_seq_len]
        """
        # Reconstruct encoder-like outputs from binary latents
        decoder_inputs = self.perceiver_ae.reconstruct_from_binary(binary_latents)
        
        # Create encoder outputs structure for BART decoder
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=decoder_inputs)
        
        # Generate using BART decoder
        generated_ids = self.bart_model.generate(
            encoder_outputs=encoder_outputs,
            **generation_kwargs
        )
        
        return generated_ids
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_binary_latents: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training the autoencoder (Stage 1).
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            labels: Target labels for training [B, seq_len]
            return_binary_latents: Whether to return binary latents
            
        Returns:
            Dictionary with loss, logits, and optionally binary latents
        """
        # BART encoder
        with torch.set_grad_enabled(not self.freeze_bart_encoder):
            encoder_outputs = self.bart_model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Compress and reconstruct via Perceiver autoencoder
        reconstructed_outputs, binary_latents = self.perceiver_ae(
            encoder_outputs.last_hidden_state,
            attention_mask
        )
        
        # Create modified encoder outputs for BART decoder
        from transformers.modeling_outputs import BaseModelOutput
        modified_encoder_outputs = BaseModelOutput(last_hidden_state=reconstructed_outputs)
        
        # BART decoder forward pass
        decoder_outputs = self.bart_model(
            encoder_outputs=modified_encoder_outputs,
            labels=labels,
            return_dict=True
        )
        
        result = {
            'loss': decoder_outputs.loss,
            'logits': decoder_outputs.logits
        }
        
        if return_binary_latents:
            result['binary_latents'] = binary_latents
        
        return result
    
    def generate_from_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 64,
        num_beams: int = 4,
        do_sample: bool = False,
        temperature: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text through binary latent compression (testing autoencoder quality).
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Tuple of (generated_ids, binary_latents)
        """
        # Encode to binary latents
        binary_latents = self.encode_text_to_binary_latents(input_ids, attention_mask)
        
        # Generate from binary latents
        generation_kwargs = {
            'max_length': max_length,
            'num_beams': num_beams,
            'do_sample': do_sample,
            'temperature': temperature,
            **kwargs
        }
        
        generated_ids = self.decode_binary_latents_to_text(
            binary_latents, 
            **generation_kwargs
        )
        
        return generated_ids, binary_latents
    
    def get_diffusion_latent(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get binary latents for diffusion (matches paper API).
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            
        Returns:
            Binary latents [B, num_encoder_latents, dim_ae]
        """
        return self.encode_text_to_binary_latents(input_ids, attention_mask)
    
    def get_decoder_input(self, diffusion_latent: torch.Tensor) -> torch.Tensor:
        """
        Convert diffusion latents to decoder input (matches paper API).
        
        Args:
            diffusion_latent: Binary latents [B, num_encoder_latents, dim_ae]
            
        Returns:
            Decoder input representations [B, num_decoder_latents, dim_lm]
        """
        return self.perceiver_ae.reconstruct_from_binary(diffusion_latent)
    
    def get_binary_latent_shapes(self) -> Tuple[int, int]:
        """
        Get the shape of binary latents for QuDiffuse compatibility.
        
        Returns:
            Tuple of (num_encoder_latents, dim_ae)
        """
        return (self.num_encoder_latents, self.dim_ae)
    
    def compute_reconstruction_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for Stage 1 training.
        
        Args:
            input_ids: Input token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]
            labels: Target labels [B, seq_len]
            
        Returns:
            Reconstruction loss
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs['loss']
    
    def set_training_stage(self, stage: int):
        """
        Set training stage (1: autoencoder, 2: diffusion).
        
        Args:
            stage: Training stage (1 or 2)
        """
        if stage == 1:
            # Stage 1: Train autoencoder, BART encoder trainable
            for param in self.bart_model.get_encoder().parameters():
                param.requires_grad = True
            for param in self.perceiver_ae.parameters():
                param.requires_grad = True
            print("🔧 Set to Stage 1: Training autoencoder + BART encoder")
            
        elif stage == 2:
            # Stage 2: Train diffusion, freeze BART encoder
            for param in self.bart_model.get_encoder().parameters():
                param.requires_grad = False
            for param in self.perceiver_ae.parameters():
                param.requires_grad = False
            print("🔧 Set to Stage 2: BART encoder + autoencoder frozen for diffusion training")
            
        else:
            raise ValueError(f"Invalid training stage: {stage}. Must be 1 or 2.")


class BARTTokenizerWrapper:
    """Wrapper for BART tokenizer with reasoning task utilities."""
    
    def __init__(self, model_name: str = "facebook/bart-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens for reasoning tasks
        special_tokens = {
            'additional_special_tokens': [
                '<reasoning>', '</reasoning>',
                '<answer>', '</answer>',
                '<step>', '</step>',
                '<arithmetic>', '</arithmetic>',
                '<spatial>', '</spatial>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        print(f"📝 BARTTokenizerWrapper initialized:")
        print(f"   Vocabulary size: {len(self.tokenizer)}")
        print(f"   Special tokens added: {len(special_tokens['additional_special_tokens'])}")
    
    def encode_reasoning_problem(self, problem: str, answer: str = None) -> Dict[str, torch.Tensor]:
        """
        Encode reasoning problem with special formatting.
        
        Args:
            problem: Input problem text
            answer: Target answer with reasoning chain
            
        Returns:
            Dictionary with input_ids, attention_mask, labels
        """
        # Format input
        input_text = f"<reasoning>{problem}</reasoning>"
        
        # Format target
        if answer is not None:
            target_text = f"<answer>{answer}</answer>"
        else:
            target_text = None
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        result = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        if target_text is not None:
            targets = self.tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            result['labels'] = targets['input_ids']
        
        return result
    
    def decode_reasoning_output(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to reasoning text.
        
        Args:
            token_ids: Generated token IDs
            
        Returns:
            Decoded reasoning text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    
    def __call__(self, *args, **kwargs):
        """Delegate to underlying tokenizer."""
        return self.tokenizer(*args, **kwargs)
    
    def __len__(self):
        """Get vocabulary size."""
        return len(self.tokenizer) 