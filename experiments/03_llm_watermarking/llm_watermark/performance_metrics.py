"""
Performance Metrics Module - Comprehensive evaluation of watermarked models
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from pathlib import Path
import math

# For evaluation metrics
try:
    from evaluate import load as load_metric
    EVALUATE_AVAILABLE = True
except ImportError:
    EVALUATE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("evaluate package not available. Some metrics will be unavailable.")

logger = logging.getLogger(__name__)


class PerformanceEvaluator:
    """Comprehensive performance evaluation for watermarked models."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize performance evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.eval_config = config['evaluation']
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['hardware']['use_cuda'] else 'cpu')
        
        # Initialize metrics
        self.metrics = {}
        if EVALUATE_AVAILABLE:
            try:
                self.perplexity_metric = load_metric("perplexity", module_type="metric")
                self.bleu_metric = load_metric("bleu")
                self.rouge_metric = load_metric("rouge")
            except Exception as e:
                logger.warning(f"Could not load some metrics: {e}")
        
        logger.info("Initialized PerformanceEvaluator")
    
    def evaluate_model_comprehensive(
        self, 
        model: nn.Module, 
        tokenizer: AutoTokenizer,
        test_dataset: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            model: The model to evaluate
            tokenizer: Tokenizer
            test_dataset: Optional test dataset. If None, loads from config.
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info("Starting comprehensive model evaluation")
        start_time = time.time()
        
        # Load test dataset if not provided
        if test_dataset is None:
            test_dataset = self._load_test_dataset(tokenizer)
        
        results = {
            'evaluation_timestamp': time.time(),
            'model_info': self._get_model_info(model),
            'dataset_info': self._get_dataset_info(test_dataset)
        }
        
        # Core language modeling metrics
        results['language_modeling'] = self._evaluate_language_modeling(model, tokenizer, test_dataset)
        
        # Text generation quality
        results['generation_quality'] = self._evaluate_generation_quality(model, tokenizer, test_dataset)
        
        # Performance benchmarks
        results['performance_benchmarks'] = self._evaluate_performance_benchmarks(model, tokenizer, test_dataset)
        
        # Model degradation analysis
        results['degradation_analysis'] = self._analyze_model_degradation(results)
        
        # Timing information
        results['evaluation_time'] = time.time() - start_time
        
        logger.info(f"Comprehensive evaluation completed in {results['evaluation_time']:.2f} seconds")
        
        return results
    
    def _load_test_dataset(self, tokenizer):
        """Load and prepare test dataset."""
        dataset_config = self.config['dataset']
        
        try:
            dataset = load_dataset(
                dataset_config['name'], 
                dataset_config.get('config_name'),
                split='test'
            )
            
            # Limit dataset size if specified
            max_samples = self.eval_config.get('max_eval_samples', -1)
            if max_samples > 0 and len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'], 
                    truncation=True, 
                    padding=True, 
                    max_length=dataset_config['max_length'],
                    return_tensors='pt'
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            logger.info(f"Loaded test dataset with {len(tokenized_dataset)} samples")
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Failed to load test dataset: {e}")
            return None
    
    def _evaluate_language_modeling(self, model: nn.Module, tokenizer, dataset) -> Dict[str, float]:
        """Evaluate core language modeling capabilities."""
        logger.info("Evaluating language modeling performance")
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        # Calculate perplexity
        with torch.no_grad():
            for i in range(0, min(len(dataset), 100), self.eval_config['eval_batch_size']):
                batch_end = min(i + self.eval_config['eval_batch_size'], len(dataset))
                batch = dataset[i:batch_end]
                
                if 'input_ids' not in batch:
                    continue
                
                input_ids = torch.tensor(batch['input_ids']).to(self.device)
                
                if input_ids.size(0) == 0:
                    continue
                
                # Forward pass
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Accumulate loss
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_tokens += input_ids.numel()
                num_batches += 1
        
        # Calculate metrics
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
        else:
            avg_loss = float('inf')
            perplexity = float('inf')
        
        return {
            'perplexity': perplexity,
            'average_loss': avg_loss,
            'total_tokens_evaluated': total_tokens,
            'num_batches': num_batches
        }
    
    def _evaluate_generation_quality(self, model: nn.Module, tokenizer, dataset) -> Dict[str, Any]:
        """Evaluate text generation quality."""
        logger.info("Evaluating text generation quality")
        
        model.eval()
        
        # Sample prompts from dataset
        sample_indices = np.random.choice(len(dataset), min(10, len(dataset)), replace=False)
        
        generation_results = {
            'samples': [],
            'average_generation_time': 0.0,
            'total_samples': 0
        }
        
        total_generation_time = 0.0
        
        with torch.no_grad():
            for idx in sample_indices:
                sample = dataset[int(idx)]
                
                if 'input_ids' not in sample:
                    continue
                
                # Use first part as prompt
                input_ids = torch.tensor(sample['input_ids'][:50]).unsqueeze(0).to(self.device)
                
                start_time = time.time()
                
                # Generate text
                try:
                    generated_ids = model.generate(
                        input_ids,
                        max_length=input_ids.size(1) + 50,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    generation_time = time.time() - start_time
                    total_generation_time += generation_time
                    
                    # Decode generated text
                    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    
                    generation_results['samples'].append({
                        'prompt': prompt_text,
                        'generated': generated_text,
                        'generation_time': generation_time
                    })
                    
                except Exception as e:
                    logger.warning(f"Generation failed for sample {idx}: {e}")
                    continue
        
        if generation_results['samples']:
            generation_results['average_generation_time'] = total_generation_time / len(generation_results['samples'])
            generation_results['total_samples'] = len(generation_results['samples'])
        
        return generation_results
    
    def _evaluate_performance_benchmarks(self, model: nn.Module, tokenizer, dataset) -> Dict[str, float]:
        """Evaluate performance benchmarks (speed, memory, etc.)."""
        logger.info("Evaluating performance benchmarks")
        
        model.eval()
        
        # Inference speed benchmark
        sample_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 512)).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids=sample_input_ids)
        
        # Benchmark inference speed
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        num_runs = 10
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids=sample_input_ids)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_runs
        tokens_per_second = sample_input_ids.size(1) / avg_inference_time
        
        # Memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        else:
            memory_allocated = 0.0
            memory_reserved = 0.0
        
        return {
            'inference_time_per_sample': avg_inference_time,
            'tokens_per_second': tokens_per_second,
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'sequence_length_tested': sample_input_ids.size(1)
        }
    
    def _analyze_model_degradation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model performance degradation due to watermarking."""
        # This would ideally compare against a baseline model
        # For now, we'll provide analysis based on absolute metrics
        
        degradation_analysis = {
            'perplexity_analysis': self._analyze_perplexity(results['language_modeling']['perplexity']),
            'performance_analysis': self._analyze_performance(results['performance_benchmarks']),
            'generation_quality_analysis': self._analyze_generation_quality(results['generation_quality'])
        }
        
        return degradation_analysis
    
    def _analyze_perplexity(self, perplexity: float) -> Dict[str, Any]:
        """Analyze perplexity score."""
        if perplexity < 10:
            quality = "excellent"
        elif perplexity < 50:
            quality = "good"
        elif perplexity < 100:
            quality = "acceptable"
        else:
            quality = "poor"
        
        return {
            'perplexity_value': perplexity,
            'quality_assessment': quality,
            'is_acceptable': perplexity < 100
        }
    
    def _analyze_performance(self, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze performance metrics."""
        tokens_per_second = performance_metrics['tokens_per_second']
        
        if tokens_per_second > 1000:
            speed_assessment = "fast"
        elif tokens_per_second > 100:
            speed_assessment = "moderate"
        else:
            speed_assessment = "slow"
        
        return {
            'speed_assessment': speed_assessment,
            'tokens_per_second': tokens_per_second,
            'memory_efficiency': performance_metrics['memory_allocated_gb'] < 8.0
        }
    
    def _analyze_generation_quality(self, generation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze generation quality."""
        if not generation_results['samples']:
            return {'quality_assessment': 'unknown', 'samples_generated': 0}
        
        avg_generation_time = generation_results['average_generation_time']
        num_samples = len(generation_results['samples'])
        
        if avg_generation_time < 1.0:
            speed_quality = "fast"
        elif avg_generation_time < 5.0:
            speed_quality = "moderate"
        else:
            speed_quality = "slow"
        
        return {
            'speed_quality': speed_quality,
            'samples_generated': num_samples,
            'average_generation_time': avg_generation_time,
            'generation_successful': num_samples > 0
        }
    
    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get model information."""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': num_params,
            'trainable_parameters': num_trainable_params,
            'model_type': type(model).__name__,
            'device': str(next(model.parameters()).device)
        }
    
    def _get_dataset_info(self, dataset) -> Dict[str, Any]:
        """Get dataset information."""
        if dataset is None:
            return {'size': 0, 'available': False}
        
        return {
            'size': len(dataset),
            'available': True,
            'features': list(dataset.features.keys()) if hasattr(dataset, 'features') else []
        }
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("WATERMARKED MODEL EVALUATION REPORT")
        report.append("=" * 60)
        
        # Model info
        model_info = results['model_info']
        report.append(f"\nModel Information:")
        report.append(f"  Type: {model_info['model_type']}")
        report.append(f"  Total Parameters: {model_info['total_parameters']:,}")
        report.append(f"  Trainable Parameters: {model_info['trainable_parameters']:,}")
        report.append(f"  Device: {model_info['device']}")
        
        # Language modeling performance
        lm_results = results['language_modeling']
        report.append(f"\nLanguage Modeling Performance:")
        report.append(f"  Perplexity: {lm_results['perplexity']:.2f}")
        report.append(f"  Average Loss: {lm_results['average_loss']:.4f}")
        report.append(f"  Tokens Evaluated: {lm_results['total_tokens_evaluated']:,}")
        
        # Performance benchmarks
        perf_results = results['performance_benchmarks']
        report.append(f"\nPerformance Benchmarks:")
        report.append(f"  Inference Time: {perf_results['inference_time_per_sample']:.4f} seconds")
        report.append(f"  Tokens/Second: {perf_results['tokens_per_second']:.1f}")
        report.append(f"  Memory Usage: {perf_results['memory_allocated_gb']:.2f} GB")
        
        # Generation quality
        gen_results = results['generation_quality']
        report.append(f"\nGeneration Quality:")
        report.append(f"  Samples Generated: {gen_results['total_samples']}")
        report.append(f"  Average Generation Time: {gen_results['average_generation_time']:.2f} seconds")
        
        # Degradation analysis
        degradation = results['degradation_analysis']
        report.append(f"\nDegradation Analysis:")
        report.append(f"  Perplexity Quality: {degradation['perplexity_analysis']['quality_assessment']}")
        report.append(f"  Speed Assessment: {degradation['performance_analysis']['speed_assessment']}")
        report.append(f"  Memory Efficient: {degradation['performance_analysis']['memory_efficiency']}")
        
        report.append(f"\nEvaluation completed in {results['evaluation_time']:.2f} seconds")
        report.append("=" * 60)
        
        return "\n".join(report)


def create_performance_evaluator(config: Dict[str, Any]) -> PerformanceEvaluator:
    """Factory function to create performance evaluator."""
    return PerformanceEvaluator(config)
