"""
Modular LLM Loader - Plug-and-play module for loading different LLMs
"""

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaForCausalLM,
    LlamaTokenizer
)
import os
from dotenv import load_dotenv
from typing import Dict, Any, Tuple, Optional
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMLoader:
    """
    Modular loader for Large Language Models.

    Responsibilities:
    - Load tokenizer and model from Hugging Face
    - Select appropriate device (cuda/mps/cpu)
    - Apply device-specific optimizations
    - Expose model info for logging/metadata
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the LLM loader with configuration.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        # Load environment variables
        try:
            load_dotenv()
        except Exception:
            pass

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.hardware_config = self.config['hardware']
        
        # Set device and configure device-specific optimizations
        self.device = self._get_device()
        self._configure_device_optimizations()
        
        # Model and tokenizer placeholders
        self.model = None
        self.tokenizer = None
        
    def _get_device(self) -> str:
        """Determine the appropriate device for model loading (cuda > mps > cpu)."""
        # Check if a specific device is forced
        if 'force_device' in self.hardware_config:
            forced_device = self.hardware_config['force_device']
            if self._is_device_available(forced_device):
                logger.info(f"Using forced device: {forced_device}")
                return forced_device
            else:
                logger.warning(f"Forced device {forced_device} not available, falling back to auto-detection")
        
        # Auto-detect best device
        device_preference = self.hardware_config.get('device', 'auto')
        
        if device_preference == 'auto':
            # Auto-select best available device
            if torch.cuda.is_available():
                logger.info("Auto-selected device: CUDA GPU")
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("Auto-selected device: Apple MPS")
                return 'mps'
            else:
                logger.info("Auto-selected device: CPU")
                return 'cpu'
        else:
            # Use specified device if available
            if self._is_device_available(device_preference):
                logger.info(f"Using specified device: {device_preference}")
                return device_preference
            else:
                logger.warning(f"Specified device {device_preference} not available, falling back to CPU")
                return 'cpu'
    
    def _is_device_available(self, device: str) -> bool:
        """Check if a specific device is available to PyTorch (feature-gated)."""
        if device == 'cuda':
            return torch.cuda.is_available()
        elif device == 'mps':
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        elif device == 'cpu':
            return True
        else:
            return False
    
    def _configure_device_optimizations(self):
        """Configure device-specific optimizations (TF32, fallback, threads)."""
        if self.device == 'cuda':
            cuda_config = self.hardware_config.get('cuda', {})
            
            # Configure TensorFloat-32
            if cuda_config.get('allow_tf32', True):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TensorFloat-32 for CUDA")
            
            # Configure cuDNN benchmarking
            if cuda_config.get('benchmark_cudnn', True):
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmarking")
            
            # Configure deterministic behavior
            if cuda_config.get('deterministic', False):
                torch.backends.cudnn.deterministic = True
                logger.info("Enabled deterministic cuDNN behavior")
                
        elif self.device == 'mps':
            mps_config = self.hardware_config.get('mps', {})
            logger.info("Using Apple MPS device")
            
            # Note: MPS-specific optimizations would go here
            if mps_config.get('fallback_to_cpu', True):
                logger.info("MPS fallback to CPU enabled for unsupported operations")
                
        elif self.device == 'cpu':
            cpu_config = self.hardware_config.get('cpu', {})
            
            # Configure CPU threads
            num_threads = cpu_config.get('num_threads', -1)
            if num_threads > 0:
                torch.set_num_threads(num_threads)
                logger.info(f"Set CPU threads to {num_threads}")
            
        logger.info(f"Device optimizations configured for: {self.device}")
    
    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load the specified model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = self.model_config['name']
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer(model_name)
        
        # Load model with appropriate configuration
        self.model = self._load_model_with_config(model_name)
        
        logger.info(f"Model loaded successfully on device: {self.device}")
        logger.info(f"Model parameters: {self._count_parameters():,}")
        
        return self.model, self.tokenizer
    
    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """Load the tokenizer and ensure a pad_token exists (use eos if missing)."""
        try:
            token = (
                os.getenv('HUGGINGFACE_ACCESS_TOKEN')
                or os.getenv('HUGGINGFACE_HUB_TOKEN')
                or os.getenv('HF_TOKEN')
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_config.get('cache_dir'),
                trust_remote_code=True,
                token=token if token else None
            )
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _load_model_with_config(self, model_name: str) -> AutoModelForCausalLM:
        """Load the model with dtype, device map, and enable gradient checkpointing if available."""
        try:
            # Determine torch dtype
            torch_dtype = getattr(torch, self.model_config.get('torch_dtype', 'float16'))
            
            # Model loading arguments
            model_kwargs = {
                'cache_dir': self.model_config.get('cache_dir'),
                'torch_dtype': torch_dtype,
                'trust_remote_code': True,
            }

            # Pass Hugging Face access token if available (for gated models like Meta Llama 3.1)
            token = (
                os.getenv('HUGGINGFACE_ACCESS_TOKEN')
                or os.getenv('HUGGINGFACE_HUB_TOKEN')
                or os.getenv('HF_TOKEN')
            )
            if token:
                model_kwargs['token'] = token
            
            # Add device map if specified
            if self.model_config.get('device_map'):
                model_kwargs['device_map'] = self.model_config['device_map']
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if device_map not used
            if not self.model_config.get('device_map'):
                model = model.to(self.device)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {}
        
        return {
            'model_name': self.model_config['name'],
            'num_parameters': self._count_parameters(),
            'device': str(next(self.model.parameters()).device),
            'dtype': str(next(self.model.parameters()).dtype),
            'model_type': type(self.model).__name__,
        }
    
    def get_target_modules(self) -> list:
        """Get the target modules for watermark embedding based on model architecture."""
        model_name_lower = self.model_config['name'].lower()
        
        if 'llama' in model_name_lower:
            # Llama architecture target modules
            return [
                'model.layers.*.self_attn.q_proj',
                'model.layers.*.self_attn.k_proj',
                'model.layers.*.self_attn.v_proj',
                'model.layers.*.self_attn.o_proj',
                'model.layers.*.mlp.gate_proj',
                'model.layers.*.mlp.up_proj',
                'model.layers.*.mlp.down_proj'
            ]
        elif 'gpt' in model_name_lower:
            # GPT architecture target modules
            return [
                'transformer.h.*.attn.c_attn',
                'transformer.h.*.attn.c_proj',
                'transformer.h.*.mlp.c_fc',
                'transformer.h.*.mlp.c_proj'
            ]
        else:
            # Generic target modules
            return self.config['watermark']['target_layers']
    
    def save_model(self, output_dir: str):
        """Save the model and tokenizer to the specified directory."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        # Save model
        self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model info
        model_info = self.get_model_info()
        with open(output_path / 'model_info.yaml', 'w') as f:
            yaml.dump(model_info, f, default_flow_style=False)
        
        logger.info("Model saved successfully")


def create_model_loader(config_path: str) -> LLMLoader:
    """Factory function to create a model loader."""
    return LLMLoader(config_path)


if __name__ == "__main__":
    # Example usage
    loader = create_model_loader("config.yaml")
    model, tokenizer = loader.load_model()
    print(loader.get_model_info())
