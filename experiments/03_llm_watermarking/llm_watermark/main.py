#!/usr/bin/env python3
"""
Main Script for LLM Watermark Fine-tuning Pipeline

This script orchestrates the entire pipeline:
1. Load configurable LLM (Llama 2 7B by default)
2. Fine-tune with custom loss function including watermark embedding
3. Measure model performance after watermarking
4. Generate comprehensive evaluation report
"""

import argparse
import logging
import sys
import time
import yaml
import json
from pathlib import Path
import torch
from datasets import load_dataset

# Import our custom modules
from .model_loader import create_model_loader
from .finetune_strategies import apply_finetune_strategy
from .watermark_embedding import create_watermark_embedder
from .custom_trainer import create_watermark_trainer
from .performance_metrics import create_performance_evaluator

# Configure logging: send to stdout and to timestamped file under configured logs dir
logs_dir = Path('./logs')
try:
    # Attempt to read logs_dir from config.yaml early (best-effort)
    with open('config.yaml', 'r') as _cf:
        _cfg = yaml.safe_load(_cf)
        logs_dir = Path(_cfg.get('output', {}).get('logs_dir', './logs'))
except Exception:
    pass

logs_dir.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_file_path = logs_dir / f'watermark_training_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM with watermark embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training and only evaluate existing model'
    )
    
    parser.add_argument(
        '--evaluate-only',
        type=str,
        default=None,
        help='Path to model for evaluation only (skip training)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_configuration(config_path: str) -> dict:
    """Load and validate configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Validate required configuration sections
        required_sections = ['model', 'watermark', 'training', 'loss', 'dataset', 'evaluation', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise


def prepare_dataset(config: dict, tokenizer):
    """Prepare training and evaluation datasets."""
    logger.info("Preparing datasets")
    
    dataset_config = config['dataset']
    
    try:
        # Load dataset
        dataset = load_dataset(
            dataset_config['name'],
            dataset_config.get('config_name'),
            cache_dir=config['model'].get('cache_dir')
        )
        
        # Prepare tokenization function
        def tokenize_function(examples):
            texts = []
            
            # Handle different dataset formats
            if 'text' in examples:
                # WikiText-style dataset
                texts = [text for text in examples['text'] if text.strip()]
            elif 'conversation' in examples:
                # WildChat-style conversational dataset
                for conversation in examples['conversation']:
                    if conversation and len(conversation) > 0:
                        # Convert conversation turns to a single text
                        conversation_text = ""
                        for turn in conversation:
                            role = turn.get('role', 'unknown')
                            content = turn.get('content', '').strip()
                            if content:
                                # Format as simple dialogue
                                conversation_text += f"{role.capitalize()}: {content}\n"
                        
                        if conversation_text.strip():
                            texts.append(conversation_text.strip())
            
            if not texts:
                return {'input_ids': [], 'attention_mask': []}
            
            # For causal language modeling, tokenize each text individually
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,  # DataCollator will handle padding
                max_length=dataset_config['max_length'],
                return_tensors=None
            )
            
            # Don't manually set labels - let DataCollatorForLanguageModeling handle it!
            # The data collator will automatically:
            # 1. Create labels from input_ids
            # 2. Shift labels by 1 position for next-token prediction  
            # 3. Set padding positions to -100
            
            return tokenized
        
        # Handle different dataset split scenarios
        if 'train' in dataset and 'validation' in dataset:
            # Dataset has both train and validation splits (e.g., WikiText)
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset['train'].column_names
            )
        else:
            # Dataset only has train split (e.g., WildChat) - create our own splits
            train_data = dataset['train'] if 'train' in dataset else dataset
            
            # Create train/validation split (90/10)
            train_size = dataset_config.get('train_size', 1000)
            eval_size = config['evaluation'].get('max_eval_samples', 200)
            total_needed = train_size + eval_size
            
            # Select subset and split
            if len(train_data) > total_needed:
                subset_data = train_data.select(range(total_needed))
            else:
                subset_data = train_data
            
            split_data = subset_data.train_test_split(test_size=eval_size, shuffle=True, seed=42)
            dataset = {'train': split_data['train'], 'validation': split_data['test']}
            
            # Tokenize the split datasets
            tokenized_dataset = dataset
            for split_name in ['train', 'validation']:
                tokenized_dataset[split_name] = dataset[split_name].map(
                    tokenize_function,
                    batched=True,
                    remove_columns=dataset[split_name].column_names
                )
        
        # Final size adjustments if needed
        train_size = dataset_config.get('train_size', -1)
        if train_size > 0 and len(tokenized_dataset['train']) > train_size:
            tokenized_dataset['train'] = tokenized_dataset['train'].select(range(train_size))
        
        eval_size = config['evaluation'].get('max_eval_samples', 200)
        if eval_size > 0 and len(tokenized_dataset['validation']) > eval_size:
            tokenized_dataset['validation'] = tokenized_dataset['validation'].select(range(eval_size))
        
        logger.info(f"Dataset prepared - Train: {len(tokenized_dataset['train'])}, Eval: {len(tokenized_dataset['validation'])}")
        
        return tokenized_dataset['train'], tokenized_dataset['validation']
        
    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise


def run_training_pipeline(config: dict, output_dir: str, config_path: str = "config.yaml"):
    """Run the complete training pipeline."""
    logger.info("Starting watermark training pipeline")
    pipeline_start_time = time.time()
    
    # Step 1: Load model and tokenizer
    logger.info("Step 1: Loading model and tokenizer")
    model_loader = create_model_loader(config_path)
    model, tokenizer = model_loader.load_model()
    
    # Log model information
    model_info = model_loader.get_model_info()
    logger.info(f"Model loaded: {model_info}")
    
    # Step 1.5: Apply finetuning strategy (standard/LoRA)
    logger.info("Step 1.5: Applying finetuning strategy")
    model, finetune_info = apply_finetune_strategy(model, config)
    logger.info(f"Finetuning strategy: {finetune_info}")
    
    # Step 2: Create watermark embedder (optional)
    logger.info("Step 2: Creating watermark embedder")
    wm_enabled = bool(config.get('watermark', {}).get('enabled', True))
    watermark_embedder = create_watermark_embedder(config) if wm_enabled else None
    
    # Log watermark information
    watermark_info = watermark_embedder.get_watermark_info() if watermark_embedder else {'enabled': False}
    if watermark_embedder:
        logger.info(f"Watermark created: {watermark_info}")
    else:
        logger.info("Watermark disabled via config; proceeding with pure fine-tuning")
    
    # Step 3: Prepare dataset
    logger.info("Step 3: Preparing dataset")
    train_dataset, eval_dataset = prepare_dataset(config, tokenizer)
    
    # Step 4: Create custom trainer
    logger.info("Step 4: Creating custom trainer with exposed loss function")
    trainer = create_watermark_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        config=config,
        watermark_embedder=watermark_embedder
    )
    
    # Step 5: Skip initial evaluation for basic demo
    logger.info("Step 5: Skipping initial evaluation for basic demo")
    initial_eval = {'note': 'Initial evaluation skipped for basic demo'}
    logger.info(f"Initial evaluation: {initial_eval}")
    
    # Step 6: Fine-tuning with watermark embedding
    logger.info("Step 6: Starting fine-tuning with watermark embedding")
    training_start_time = time.time()
    
    # Train the model
    train_result = trainer.train()
    
    training_time = time.time() - training_start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Save per-epoch CSV with epoch/shard/lr/watermark_rate/task_loss (under ./output or configured csv_dir)
    try:
        csv_path = getattr(trainer, 'save_epoch_csv', None)
        if callable(csv_path):
            csv_dir = config.get('output', {}).get('csv_dir', './output')
            trainer.save_epoch_csv(csv_dir)
    except Exception as e:
        logger.warning(f"Failed to write epoch CSV: {e}")
    
    # Step 7: Skip final evaluation for basic demo (optional)
    logger.info("Step 7: Skipping final evaluation for basic demo")
    final_eval = {'note': 'Final evaluation skipped for basic demo'}
    logger.info(f"Final evaluation: {final_eval}")
    
    # Step 8: Watermark verification
    logger.info("Step 8: Performing final watermark verification")
    watermark_stats = trainer.get_watermark_stats()
    logger.info(f"Watermark verification: Success = {watermark_stats['success_achieved']}")
    logger.info(f"Final match rate: {watermark_stats['final_match_rate']:.6f}")
    
    # Step 9: Save model and results
    logger.info("Step 9: Saving model and results")
    trainer.save_model(output_dir)
    
    # Step 10: Skip comprehensive evaluation for basic demo
    logger.info("Step 10: Skipping comprehensive performance evaluation for basic demo")
    performance_results = {'note': 'Comprehensive evaluation skipped for basic demo'}
    
    # Save performance results
    performance_results_path = Path(output_dir) / "performance_evaluation.json"
    with open(performance_results_path, 'w') as f:
        import json
        json.dump(performance_results, f, indent=2)
    
    # Generate evaluation report
    evaluation_report = f"Evaluation Report:\n{performance_results['note']}"
    report_path = Path(output_dir) / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(evaluation_report)
    
    logger.info("Evaluation report generated")
    print(evaluation_report)
    
    # Step 11: Summary results
    pipeline_time = time.time() - pipeline_start_time
    
    summary_results = {
        'pipeline_completion_time': pipeline_time,
        'training_time': training_time,
        'train_result': train_result,  # Add training results for summary
        'model_info': model_info,
        'watermark_info': watermark_info,
        'watermark_stats': watermark_stats,
        'initial_evaluation': initial_eval,
        'final_evaluation': final_eval,
        'final_verification': watermark_stats,  # Add final verification
        'performance_evaluation': performance_results,
        'dataset_info': {
            'train_size': len(train_dataset) if train_dataset is not None else config['dataset'].get('train_size', 'N/A'),
            'eval_size': len(eval_dataset) if eval_dataset is not None else config['evaluation'].get('max_eval_samples', 'N/A')
        },
        'config': config
    }
    
    # Save summary
    summary_path = Path(output_dir) / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2, default=str)
    
    logger.info(f"Pipeline completed successfully in {pipeline_time:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")
    
    # Generate beautiful summary
    generate_beautiful_summary(summary_results, pipeline_time, config)
    
    return summary_results


def generate_beautiful_summary(results: dict, total_time: float, config: dict):
    """Generate a beautiful summary of the training results."""
    
    def print_and_log(message):
        """Helper function to both print to console and log to file."""
        print(message)
        logger.info(message)
    
    print_and_log("\n" + "="*80)
    print_and_log("WATERMARK FINE-TUNING EXPERIMENT COMPLETE!")
    print_and_log("="*80)
    
    # Training Performance Section
    print_and_log("\nTRAINING PERFORMANCE:")
    print_and_log("─" * 40)
    training_result = results.get('train_result', results.get('training_result', {}))
    final_verification = results.get('final_verification', {})
    
    # Handle TrainOutput object or dictionary
    if hasattr(training_result, 'metrics'):
        # TrainOutput object
        metrics = getattr(training_result, 'metrics', {})
        epochs_completed = getattr(training_result, 'epoch', metrics.get('epoch', 'N/A'))
        training_time = metrics.get('train_runtime', 'N/A')
        samples_per_sec = metrics.get('train_samples_per_second', 'N/A')
        train_loss = metrics.get('train_loss', 'N/A')
    else:
        # Dictionary
        epochs_completed = training_result.get('epoch', 'N/A')
        training_time = training_result.get('train_runtime', 'N/A')
        samples_per_sec = training_result.get('train_samples_per_second', 'N/A')
        train_loss = training_result.get('train_loss', 'N/A')
        
    if isinstance(training_time, (int, float)):
        training_time = f"{training_time:.2f}s"
    if isinstance(samples_per_sec, (int, float)):
        samples_per_sec = f"{samples_per_sec:.2f}"
    if isinstance(train_loss, (int, float)):
        train_loss = f"{train_loss:.4f}"
    
    print_and_log(f"  • Total Pipeline Time:     {total_time:.2f} seconds")
    # Normalize N/A fields by falling back to values we track elsewhere
    if training_time == 'N/A' and isinstance(results.get('training_time', None), (int, float)):
        training_time = f"{results['training_time']:.2f}s"
    if epochs_completed == 'N/A' and hasattr(training_result, 'epoch'):
        epochs_completed = getattr(training_result, 'epoch', 'N/A')
    print_and_log(f"  • Training Time:           {training_time}")
    print_and_log(f"  • Epochs Completed:        {epochs_completed}")
    print_and_log(f"  • Training Loss:           {train_loss}")
    print_and_log(f"  • Samples/Second:          {samples_per_sec}")
    
    # Watermark Results Section (match-rate/BER)
    print_and_log("\nWATERMARK RESULTS:")
    print_and_log("─" * 40)
    # Extract verification results from nested structure
    current_verification = final_verification.get('current_verification', final_verification)
    watermark_success = current_verification.get('success', final_verification.get('success_achieved', False))
    match_rate = current_verification.get('overall_match_rate', final_verification.get('final_match_rate', None))
    ber = current_verification.get('overall_ber', final_verification.get('final_ber', None))
    threshold = config['watermark'].get('match_success_threshold', None)
    bit_length = config['watermark']['bit_length']
    
    if match_rate is not None and ber is not None and threshold is not None:
        print_and_log(f"  • Match Rate:              {match_rate:.6f}")
        print_and_log(f"  • BER (1 - match):         {ber:.6f}")
        print_and_log(f"  • Success Threshold:       {float(threshold):.2f}")
    else:
        print_and_log("  • Match Rate:              N/A")
        print_and_log("  • BER:                     N/A")
        print_and_log("  • Success Threshold:       N/A")
    print_and_log(f"  • Embedding Success:       {watermark_success}")
    print_and_log(f"  • Watermark Bits:          {bit_length}-bit")
    print_and_log(f"  • Target Layers:           {', '.join(config['watermark']['target_layers'])}")
    
    # Model Information Section
    print_and_log("\nMODEL INFORMATION:")
    print_and_log("─" * 40)
    model_info = results.get('model_info', {})
    model_name = model_info.get('model_name', config['model']['name'])
    num_params = model_info.get('num_parameters', 'N/A')
    if isinstance(num_params, int):
        num_params = f"{num_params:,}"
    device = model_info.get('device', 'N/A')
    dtype = model_info.get('dtype', 'N/A')
    
    print_and_log(f"  • Model:                   {model_name}")
    print_and_log(f"  • Parameters:              {num_params}")
    print_and_log(f"  • Device:                  {device}")
    print_and_log(f"  • Precision:               {dtype}")
    
    # Configuration Summary
    print_and_log("\nCONFIGURATION SUMMARY:")
    print_and_log("─" * 40)
    dataset_info = results.get('dataset_info', {})
    train_samples = dataset_info.get('train_size', config['dataset'].get('train_size', 'N/A'))
    # Use actual eval dataset length if we created one
    eval_samples = dataset_info.get('eval_size', config['evaluation'].get('max_eval_samples', 'N/A'))
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    
    print_and_log(f"  • Training Samples:        {train_samples}")
    print_and_log(f"  • Evaluation Samples:      {eval_samples}")
    print_and_log(f"  • Batch Size:              {batch_size}")
    print_and_log(f"  • Learning Rate:           {learning_rate}")
    print_and_log(f"  • Max Epochs:              {config['training']['max_epochs']}")
    print_and_log(f"  • Min Epochs:              {config['training']['min_epochs']}")
    
    # Shard Information (for dual-purpose training)
    shard_info = final_verification.get('shard_info', {})
    if shard_info.get('is_dual_training', False):
        print_and_log("\nSHARD-BASED TRAINING:")
        print_and_log("─" * 40)
        dual_training_mode = "Watermark + Language Learning" if config['loss']['base_loss_weight'] > 0 else "Pure Watermark Embedding"
        print_and_log(f"  • Training Mode:           {dual_training_mode}")
        print_and_log(f"  • Base Loss Weight:        {config['loss']['base_loss_weight']}")
        print_and_log(f"  • Watermark Loss Weight:   {config['loss']['watermark_loss_weight']}")
        print_and_log(f"  • Total Shards Completed:  {shard_info.get('total_shards_completed', 0)}")
        print_and_log(f"  • Current Shard:           {shard_info.get('current_shard', 1)}")
        print_and_log(f"  • Max Shards Configured:   {shard_info.get('shard_config', {}).get('max_shards', 'N/A')}")
        
        # Show shard history if available
        shard_history = shard_info.get('shard_history', [])
        if shard_history:
            print_and_log("  • Shard History:")
            for shard in shard_history:
                duration = shard.get('epochs_duration', 'N/A')
                success_rate = shard.get('watermark_success_rate', 'N/A')
                language_loss = shard.get('language_loss', 'N/A')
                if isinstance(success_rate, (int, float)):
                    success_rate = f"{success_rate:.3f}"
                if isinstance(language_loss, (int, float)):
                    language_loss = f"{language_loss:.3f}"
                print_and_log(f"    - Shard {shard['shard_number']}: {duration} epochs, watermark={success_rate}, lang_loss={language_loss}")
    else:
        print_and_log("\nTRAINING MODE:")
        print_and_log("─" * 40)
        training_mode = "Watermark + Language Learning" if config['loss']['base_loss_weight'] > 0 else "Pure Watermark Embedding"
        print_and_log(f"  • Mode:                    {training_mode}")
        print_and_log(f"  • Base Loss Weight:        {config['loss']['base_loss_weight']}")
        print_and_log(f"  • Watermark Loss Weight:   {config['loss']['watermark_loss_weight']}")
        if config['loss']['base_loss_weight'] > 0:
            print_and_log("  • Shard Training:          Disabled (single-shot training)")
    
    # Performance Verdict
    print_and_log("\nEXPERIMENT VERDICT:")
    print_and_log("─" * 40)
    if watermark_success:
        print_and_log("  SUCCESS! Watermark successfully embedded by match-rate criterion!")
    else:
        print_and_log("  Not yet successful by match-rate criterion.")
        print_and_log("     Increase data/epochs or watermark loss strength.")
    
    # Next Steps
    print_and_log("\nSUGGESTED NEXT STEPS:")
    print_and_log("─" * 40)
    if not watermark_success:
        print_and_log("  1. Increase training data size in config.yaml")
        print_and_log("  2. Allow more max_epochs for longer training")
        print_and_log("  3. Adjust watermark_loss_weight in loss configuration")
        print_and_log("  4. Try different embedding_strength values")
    else:
        print_and_log("  1. Test watermark detection on generated text")
        print_and_log("  2. Scale up to larger models (Llama-2-7b-hf)")
        print_and_log("  3. Experiment with different bit lengths")
        print_and_log("  4. Evaluate robustness against attacks")
    
    print_and_log("\n" + "="*80)
    print_and_log("Ready for your next experiment!")
    print_and_log("="*80 + "\n")


def run_evaluation_only(model_path: str, config: dict):
    """Run evaluation only on an existing model."""
    logger.info(f"Running evaluation only on model: {model_path}")
    
    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Create evaluator
    evaluator = create_performance_evaluator(config)
    
    # Run evaluation
    results = evaluator.evaluate_model_comprehensive(model, tokenizer)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    print_and_log(report)
    
    # Save results
    output_dir = Path(model_path).parent / "evaluation_only"
    output_dir.mkdir(exist_ok=True)
    
    evaluator.save_evaluation_results(results, str(output_dir / "evaluation_results.json"))
    
    with open(output_dir / "evaluation_report.txt", 'w') as f:
        f.write(report)
    
    return results


def main():
    """Main function."""
    global args
    args = parse_arguments()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = load_configuration(args.config)
        
        # Override output directory if specified
        if args.output_dir:
            config['output']['model_dir'] = args.output_dir
        # Finetune strategy is controlled solely via config.yaml now
        
        output_dir = config['output']['model_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save configuration to output directory
        config_output_path = Path(output_dir) / "config_used.yaml"
        with open(config_output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        if args.evaluate_only:
            # Evaluation only mode
            results = run_evaluation_only(args.evaluate_only, config)
        else:
            # Full training pipeline
            results = run_training_pipeline(config, output_dir, args.config)
        
        logger.info("Process completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Process failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
