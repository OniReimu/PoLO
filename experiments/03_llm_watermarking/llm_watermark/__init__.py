"""
LLM Watermark Fine-tuning Pipeline

A comprehensive system for fine-tuning Large Language Models (LLMs) with 
embedding watermarks at the parameter level.
"""

__version__ = "0.1.0"
__author__ = "Anonymous Authors"

# Import main components for easy access
from .model_loader import LLMLoader, create_model_loader
from .watermark_embedding import (
    WatermarkEmbedder, 
    WatermarkLoss, 
    create_watermark_embedder,
    create_watermark_loss
)
from .custom_trainer import WatermarkTrainer, create_watermark_trainer
from .performance_metrics import PerformanceEvaluator, create_performance_evaluator

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Core classes
    "LLMLoader",
    "WatermarkEmbedder",
    "WatermarkLoss", 
    "WatermarkTrainer",
    "PerformanceEvaluator",
    
    # Factory functions
    "create_model_loader",
    "create_watermark_embedder",
    "create_watermark_loss",
    "create_watermark_trainer", 
    "create_performance_evaluator",
]
