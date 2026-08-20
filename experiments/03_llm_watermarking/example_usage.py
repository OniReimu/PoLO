#!/usr/bin/env python3
"""
Example Usage Script - Demonstrates how to use the watermark pipeline programmatically
"""

import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_config():
    """Create an example configuration for demonstration."""
    config = {
        'model': {
            'name': 'meta-llama/Llama-2-7b-hf',
            'cache_dir': './models',
            'torch_dtype': 'float16',
            'device_map': 'auto'
        },
        'watermark': {
            'bit_length': 128,
            'embedding_strength': 0.001,
            'match_success_threshold': 0.9,
            'target_layers': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        },
        'training': {
            'epochs': 2,  # Reduced for quick example
            'learning_rate': 5e-5,
            'batch_size': 2,  # Reduced for memory
            'gradient_accumulation_steps': 4,
            'warmup_steps': 50,
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
            'save_strategy': 'epoch',
            'evaluation_strategy': 'epoch',
            'logging_steps': 10
        },
        'loss': {
            'base_loss_weight': 1.0,
            'watermark_loss_weight': 0.1
        },
        'dataset': {
            'name': 'wikitext',
            'config_name': 'wikitext-2-raw-v1',
            'max_length': 256,  # Reduced for quick example
            'train_size': 100   # Very small for demo
        },
        'evaluation': {
            'metrics': ['perplexity'],
            'eval_batch_size': 4,
            'max_eval_samples': 50  # Small for demo
        },
        'output': {
            'model_dir': './example_output',
            'logs_dir': './logs',
            'results_file': './results.json'
        },
        'hardware': {
            'use_cuda': True,
            'mixed_precision': True,
            'dataloader_num_workers': 2
        }
    }
    
    return config


def example_basic_usage():
    """Example 1: Basic usage with default configuration."""
    logger.info("=== Example 1: Basic Usage ===")
    
    # Create example config
    config = create_example_config()
    
    # Save config
    config_path = 'example_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Example configuration saved to {config_path}")
    
    # Import and run pipeline components
    try:
        from llm_watermark.model_loader import create_model_loader
        from llm_watermark.watermark_embedding import create_watermark_embedder
        
        # Load model
        loader = create_model_loader(config_path)
        logger.info("Model loader created successfully")
        
        # Create watermark embedder
        embedder = create_watermark_embedder(config)
        logger.info("Watermark embedder created successfully")
        
        # Print watermark info
        watermark_info = embedder.get_watermark_info()
        logger.info(f"Watermark info: {watermark_info}")
        
        logger.info("Basic setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")


def example_custom_watermark():
    """Example 2: Custom watermark configuration."""
    logger.info("=== Example 2: Custom Watermark ===")
    
    config = create_example_config()
    
    # Customize watermark settings
    config['watermark'].update({
        'bit_length': 64,  # Shorter watermark
        'embedding_strength': 0.01,  # Stronger embedding
        'match_success_threshold': 0.85,  # Lower threshold
        'target_layers': ['q_proj', 'v_proj']  # Fewer layers
    })
    
    # Customize loss weights
    config['loss'].update({
        'watermark_loss_weight': 0.2  # Higher watermark weight
    })
    
    try:
        from llm_watermark.watermark_embedding import create_watermark_embedder
        
        embedder = create_watermark_embedder(config)
        watermark_info = embedder.get_watermark_info()
        
        logger.info(f"Custom watermark created:")
        logger.info(f"  Bit length: {watermark_info['bit_length']}")
        logger.info(f"  Embedding strength: {watermark_info['embedding_strength']}")
        logger.info(f"  Match success threshold: {watermark_info['match_success_threshold']}")
        
    except Exception as e:
        logger.error(f"Custom watermark example failed: {e}")


def example_different_model():
    """Example 3: Using a different model."""
    logger.info("=== Example 3: Different Model ===")
    
    config = create_example_config()
    
    # Use a smaller model for demonstration
    config['model']['name'] = 'gpt2'  # Much smaller model
    config['dataset']['train_size'] = 50  # Even smaller dataset
    
    try:
        from llm_watermark.model_loader import create_model_loader
        
        config_path = 'gpt2_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        loader = create_model_loader(config_path)
        model_info = {
            'model_name': config['model']['name'],
            'config_path': config_path
        }
        
        logger.info(f"Alternative model configuration:")
        logger.info(f"  Model: {model_info['model_name']}")
        logger.info(f"  Config saved to: {model_info['config_path']}")
        
    except Exception as e:
        logger.error(f"Different model example failed: {e}")


def example_evaluation_only():
    """Example 4: Evaluation-only mode."""
    logger.info("=== Example 4: Evaluation Only ===")
    
    # This example shows how to evaluate an existing model
    # (assuming you have a pre-trained model)
    
    config = create_example_config()
    
    try:
        from llm_watermark.performance_metrics import create_performance_evaluator
        
        evaluator = create_performance_evaluator(config)
        
        logger.info("Performance evaluator created for evaluation-only mode")
        logger.info("To use: python main.py --evaluate-only /path/to/model")
        
        # Example of what evaluation would measure
        evaluation_aspects = [
            "Language modeling perplexity",
            "Text generation quality",
            "Inference speed (tokens/second)",
            "Memory usage",
            "Watermark verification",
            "Performance degradation analysis"
        ]
        
        logger.info("Evaluation will measure:")
        for aspect in evaluation_aspects:
            logger.info(f"  - {aspect}")
            
    except Exception as e:
        logger.error(f"Evaluation example failed: {e}")


def main():
    """Run all examples."""
    logger.info("Running LLM Watermark Pipeline Examples")
    logger.info("=" * 50)
    
    # Run examples
    example_basic_usage()
    print()
    
    example_custom_watermark()
    print()
    
    example_different_model()
    print()
    
    example_evaluation_only()
    print()
    
    logger.info("=" * 50)
    logger.info("Examples completed!")
    logger.info("\nTo run the full pipeline:")
    logger.info("  # Using uv (recommended):")
    logger.info("  uv run python -m llm_watermark.main")
    logger.info("  # Using pip:")
    logger.info("  python -m llm_watermark.main")
    logger.info("\nTo customize settings:")
    logger.info("  1. Edit config.yaml (main configuration file)")
    logger.info("  2. Check examples/ directory for reference settings")
    logger.info("  3. Copy useful settings from examples to config.yaml")
    logger.info("\nFor help:")
    logger.info("  uv run python -m llm_watermark.main --help")
    logger.info("\nTo install with uv:")
    logger.info("  uv sync  # Install dependencies")
    logger.info("  uv sync --extra performance  # With performance extras")


if __name__ == "__main__":
    main()
