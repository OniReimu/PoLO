"""
Watermark Embedding Module - Implements 128-bit watermark embedding at parameter level
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import hashlib
import random
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class WatermarkEmbedder:
    """
    Implements watermark embedding in neural network parameters.

    Workflow:
    - Generate a binary watermark (length configurable)
    - Select deterministic parameter positions per layer (position matrix)
    - During training, add an auxiliary loss that nudges selected params
      toward ±embedding_strength depending on bit value
    - After training, verify via bit match rate (BER) using Hamming distance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize watermark embedder.
        
        Args:
            config: Configuration dictionary containing watermark settings
        """
        self.config = config
        self.watermark_config = config['watermark']
        
        # Watermark properties
        self.bit_length = self.watermark_config['bit_length']
        self.embedding_strength = self.watermark_config['embedding_strength']
        self.target_layers = self.watermark_config['target_layers']
        # Match-rate threshold for watermark success (required)
        self.match_success_threshold = self.watermark_config.get('match_success_threshold')
        if self.match_success_threshold is None:
            raise ValueError("match_success_threshold must be specified in watermark config")
        
        # Generate or load watermark
        self.watermark_bits = self._generate_watermark()
        
        # Track embedded layers
        self.embedded_layers = {}
        self.embedding_masks = {}
        
        logger.info(f"Initialized watermark embedder with {self.bit_length}-bit watermark")
    
    def _generate_watermark(self, model=None) -> torch.Tensor:
        """Generate a pseudo-random watermark sequence.
        
        Args:
            model: Optional model to use for generating seed from weights.
                  If None, uses default seed for initial shard.
        """
        if model is not None:
            # Generate seed from model weights hash (for shard n+1 based on shard n)
            seed = self._generate_seed_from_model_weights(model)
            logger.info(f"Generated watermark seed from model weights: {seed}")
        else:
            # Use default seed for initial shard
            seed = self.watermark_config.get('seed', 42)
            logger.info(f"Using default watermark seed: {seed}")
            
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random bits
        bits = torch.randint(0, 2, (self.bit_length,), dtype=torch.float32)
        
        # Update the stored watermark bits for verification
        self.watermark_bits = bits
        
        logger.info(f"Generated {self.bit_length}-bit watermark: {bits[:8]}... (showing first 8 bits)")
        return bits
    
    def _generate_seed_from_model_weights(self, model) -> int:
        """Generate a deterministic seed from model weights hash.
        
        This creates a unique seed for each shard based on the model state
        at the end of the previous shard, ensuring different watermarks per shard.
        
        Args:
            model: The trained model whose weights will be hashed
            
        Returns:
            int: Deterministic seed derived from model weights
        """
        import hashlib
        
        # Collect weights from target layers
        target_weights = []
        for name, param in model.named_parameters():
            # Only use target layers for hash to focus on watermarked parameters
            layer_base = name.split('.weight')[0].split('.bias')[0]
            if any(target in layer_base for target in self.target_layers):
                # Convert to CPU and detach for hashing
                weight_data = param.detach().cpu().numpy()
                target_weights.append(weight_data.flatten())
        
        if not target_weights:
            # Fallback: use all parameters if no target layers found
            logger.warning("No target layer weights found for hash, using all parameters")
            for name, param in model.named_parameters():
                weight_data = param.detach().cpu().numpy()
                target_weights.append(weight_data.flatten())
        
        # Concatenate all weights and create hash
        all_weights = np.concatenate(target_weights)
        
        # Create hash from weights (use a subset for efficiency)
        sample_size = min(10000, len(all_weights))  # Sample for large models
        weight_sample = all_weights[::len(all_weights)//sample_size][:sample_size]
        
        # Convert to bytes and hash
        weight_bytes = weight_sample.astype(np.float32).tobytes()
        hash_obj = hashlib.sha256(weight_bytes)
        hash_hex = hash_obj.hexdigest()
        
        # Convert hash to integer seed (use first 8 hex chars)
        seed = int(hash_hex[:8], 16)
        
        logger.info(f"Generated seed {seed} from {len(target_weights)} target layer weights")
        return seed
    
    def _get_embedding_positions(self, param_shape: Tuple[int, ...], layer_name: str) -> torch.Tensor:
        """Return reproducible positions in a parameter tensor for watermark bits."""
        # Create reproducible random positions based on layer name
        layer_hash = hashlib.md5(layer_name.encode()).hexdigest()
        seed = int(layer_hash[:8], 16)
        
        torch.manual_seed(seed)
        
        total_params = np.prod(param_shape)
        
        if total_params < self.bit_length:
            logger.warning(f"Layer {layer_name} has fewer parameters ({total_params}) than watermark bits ({self.bit_length})")
            # Use all available positions
            positions = torch.arange(total_params)
        else:
            # Sample random positions without replacement
            positions = torch.randperm(total_params)[:self.bit_length]
        
        return positions
    
    def embed_watermark_in_layer(self, param: torch.nn.Parameter, layer_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed watermark in a specific layer's parameters.
        
        Args:
            param: Parameter tensor to embed watermark in
            layer_name: Name of the layer
            
        Returns:
            Tuple of (watermark_loss, embedding_mask)
        """
        param_flat = param.view(-1)
        positions = self._get_embedding_positions(param.shape, layer_name)
        
        # Create embedding mask
        mask = torch.zeros_like(param_flat)
        mask[positions] = 1.0
        
        # Store mask for later verification
        self.embedding_masks[layer_name] = mask.view(param.shape)
        
        # Calculate watermark loss - encourage parameter values to match watermark bits
        watermark_values = self.watermark_bits[:len(positions)]
        target_values = (watermark_values * 2 - 1) * self.embedding_strength  # Convert 0,1 to -strength,+strength
        
        if positions.device != param.device:
            positions = positions.to(param.device)
            target_values = target_values.to(param.device)
        
        # Watermark loss: stable MSE between selected parameters and target watermark values
        # Cast to float32 for numerical stability on MPS/FP16
        selected_params = param_flat[positions].float()
        target_values = target_values.float()
        watermark_loss = torch.nn.functional.mse_loss(selected_params, target_values)
        
        return watermark_loss, mask.view(param.shape)
    
    def compute_watermark_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute watermark loss for all target layers in the model.
        
        Args:
            model: Neural network model
            
        Returns:
            Total watermark loss
        """
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        loss_count = 0
        
        for name, param in model.named_parameters():
            if self._should_embed_in_layer(name) and param.requires_grad:
                watermark_loss, _ = self.embed_watermark_in_layer(param, name)
                total_loss += watermark_loss
                loss_count += 1
        
        if loss_count > 0:
            return total_loss / loss_count
        else:
            return total_loss
    
    def _should_embed_in_layer(self, layer_name: str) -> bool:
        """Check if watermark should be embedded in this layer."""
        target_patterns = self.watermark_config['target_layers']
        
        for pattern in target_patterns:
            if pattern in layer_name:
                return True
        return False
    
    def verify_watermark(self, model: nn.Module) -> Dict[str, float]:
        """
        Verify watermark presence in the model.
        
        Args:
            model: Neural network model to verify
            
        Returns:
            Dictionary with verification results
        """
        verification_results = {}
        total_bits = 0
        total_matches = 0
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if self._should_embed_in_layer(name):
                    # Bit extraction and match rate calculation
                    param_flat = param.view(-1)
                    positions = self._get_embedding_positions(param.shape, name)
                    if positions.device != param.device:
                        positions = positions.to(param.device)
                    selected_params = param_flat[positions]
                    predicted_bits = (selected_params >= 0).to(torch.float32).cpu()
                    expected_bits = self.watermark_bits[:len(positions)].cpu()
                    matches = (predicted_bits == expected_bits).to(torch.float32).sum().item()
                    match_rate = matches / float(len(positions)) if len(positions) > 0 else 0.0
                    verification_results.setdefault('match_rate_per_layer', {})[name] = match_rate
                    total_bits += int(len(positions))
                    total_matches += int(matches)
        
        overall_match_rate = (total_matches / float(total_bits)) if total_bits > 0 else 0.0
        overall_ber = 1.0 - overall_match_rate

        verification_results['overall_match_rate'] = overall_match_rate
        verification_results['overall_ber'] = overall_ber
        verification_results['success'] = overall_match_rate >= float(self.match_success_threshold)
        
        logger.info(f"Watermark verification - Match Rate: {overall_match_rate:.6f} (Threshold: {self.match_success_threshold:.2f})")
        logger.info(f"Watermark verification - BER: {overall_ber:.6f}")
        logger.info(f"Watermark successfully embedded: {verification_results['success']}")
        
        return verification_results
    

    
    def get_watermark_info(self) -> Dict:
        """Get information about the watermark."""
        return {
            'bit_length': self.bit_length,
            'embedding_strength': self.embedding_strength,
            'match_success_threshold': self.match_success_threshold,

            'target_layers': self.watermark_config['target_layers'],
            'watermark_hash': hashlib.md5(self.watermark_bits.numpy().tobytes()).hexdigest()
        }
    
    def save_watermark(self, filepath: str):
        """Save watermark information to file."""
        watermark_data = {
            'watermark_bits': self.watermark_bits.tolist(),
            'config': self.watermark_config,
            'info': self.get_watermark_info()
        }
        
        torch.save(watermark_data, filepath)
        logger.info(f"Watermark saved to {filepath}")
    
    def load_watermark(self, filepath: str):
        """Load watermark from file."""
        watermark_data = torch.load(filepath)
        self.watermark_bits = torch.tensor(watermark_data['watermark_bits'])
        logger.info(f"Watermark loaded from {filepath}")


class WatermarkLoss(nn.Module):
    """Custom loss module that combines base loss with (optional) watermark loss."""
    
    def __init__(self, embedder: Optional[WatermarkEmbedder], loss_config: Dict):
        super().__init__()
        self.embedder = embedder
        self.watermark_enabled = embedder is not None
        self.base_loss_weight = loss_config['base_loss_weight']
        self.watermark_loss_weight = loss_config['watermark_loss_weight']
        
        # Base language modeling loss
        self.base_loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, model: nn.Module, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.
        
        Args:
            model: The neural network model
            logits: Model output logits
            labels: Target labels
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Base language modeling loss (causal LM: shift logits/labels)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        base_loss = self.base_loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Watermark embedding loss (optional)
        if self.watermark_enabled:
            watermark_loss = self.embedder.compute_watermark_loss(model)
        else:
            watermark_loss = shift_logits.new_tensor(0.0)
        
        # Combined loss
        total_loss = (self.base_loss_weight * base_loss + 
                     self.watermark_loss_weight * watermark_loss)
        
        loss_components = {
            'total_loss': total_loss.item(),
            'base_loss': base_loss.item(),
            'watermark_loss': watermark_loss.item() if isinstance(watermark_loss, torch.Tensor) else float(watermark_loss),
            'base_loss_weight': self.base_loss_weight,
            'watermark_loss_weight': self.watermark_loss_weight
        }
        
        return total_loss, loss_components


def create_watermark_embedder(config: Dict) -> WatermarkEmbedder:
    """Factory function to create watermark embedder."""
    return WatermarkEmbedder(config)


def create_watermark_loss(embedder: WatermarkEmbedder, loss_config: Dict) -> WatermarkLoss:
    """Factory function to create watermark loss."""
    return WatermarkLoss(embedder, loss_config)
