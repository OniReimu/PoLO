"""
Custom Trainer with Exposed Loss Function and Watermark Integration
"""

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from transformers.trainer_utils import EvalPrediction
from typing import Dict, Any, Optional, Tuple, List
import csv
import logging
import time
import json
from pathlib import Path
import numpy as np

from .watermark_embedding import WatermarkEmbedder, WatermarkLoss

logger = logging.getLogger(__name__)


class WatermarkEarlyStoppingCallback(TrainerCallback):
    """
    Custom callback for watermark-based early stopping.
    
    This callback monitors watermark match rate after each epoch and stops training
    when the success threshold is reached or early stopping criteria are met.
    """
    
    def __init__(self, watermark_trainer):
        self.watermark_trainer = watermark_trainer
        
    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of each epoch."""
        logger.info(f"CALLBACK: WatermarkEarlyStoppingCallback epoch_begin for epoch {state.epoch}")
        self.watermark_trainer._on_epoch_begin_logic(args, state, control, model, **kwargs)
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch - triggers watermark verification and early stopping."""
        logger.info(f"CALLBACK: WatermarkEarlyStoppingCallback triggered for epoch {state.epoch}")
        
        # Delegate to the trainer's epoch end logic
        self.watermark_trainer._on_epoch_end_logic(args, state, control, model, **kwargs)


class WatermarkTrainer(Trainer):
    """
    Custom trainer integrating watermark embedding with an exposed loss function.

    - compute_loss: combines base LM loss with watermark auxiliary loss
    - on_epoch_end: verifies watermark and supports early stopping
    - Metrics are recorded into training_metrics for analysis and saved at the end
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset,
        eval_dataset,
        tokenizer,
        watermark_embedder: WatermarkEmbedder,
        config: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize the watermark trainer.
        
        Args:
            model: The neural network model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            watermark_embedder: Watermark embedding module
            config: Configuration dictionary
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
        
        self.watermark_embedder = watermark_embedder
        self.watermark_enabled = watermark_embedder is not None
        self.config = config
        self.loss_config = config['loss']
        
        # Initialize custom loss function (embedder may be None when watermark disabled)
        self.watermark_loss_fn = WatermarkLoss(watermark_embedder, self.loss_config)
        
        # Training metrics tracking
        self.training_metrics = {
            'epoch_times': [],
            'loss_components': [],
            'watermark_verification': [],
            'performance_metrics': []
        }
        # CSV epoch-level records
        self.csv_records: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.start_time = None
        self.epoch_start_time = None
        
        # Dynamic stopping tracking
        self.best_metric_value = 0.0            # best match-rate
        self.epochs_without_improvement = 0
        self.early_stopping_config = config['training'].get('early_stopping', {})
        self.min_epochs = config['training'].get('min_epochs', 1)
        # Match-rate threshold for watermark success (required)
        self.match_success_threshold = config['watermark'].get('match_success_threshold')
        if self.match_success_threshold is None:
            raise ValueError("match_success_threshold must be specified in watermark config")
            
        # Shard tracking for dual-purpose training
        self.shard_config = config['training'].get('shard_training', {})
        self.is_dual_training = config['loss']['base_loss_weight'] > 0.0
        self.current_shard = 1
        self.total_shards = 0
        self.shard_history = []
        self.shard_start_epoch = 1  # Initialize shard start epoch
        
        # Language task tracking
        self.language_performance_history = []
        self.best_language_loss = float('inf')
        self.language_epochs_without_improvement = 0
        self.language_success_threshold = self.shard_config.get('language_success_threshold', 3.5)
        self.language_eval_frequency = self.shard_config.get('language_eval_frequency', 1)
        self.language_task_completed = False
        
        logger.info("Initialized WatermarkTrainer with exposed loss function")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute total loss = base LM loss + watermark auxiliary loss."""
        labels = inputs.get("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute combined loss using custom loss function
        total_loss, loss_components = self.watermark_loss_fn(model, logits, labels)
        
        # Store loss components for logging
        if hasattr(self, '_current_loss_components'):
            self._current_loss_components.update(loss_components)
        else:
            self._current_loss_components = loss_components
        
        if return_outputs:
            return total_loss, outputs
        return total_loss
    
    def training_step(self, model: nn.Module, inputs: Dict[str, torch.Tensor], num_items_in_batch=None) -> torch.Tensor:
        """
        Perform a training step with custom loss computation.
        Override just to use our custom loss, but delegate everything else to parent.
        
        Args:
            model: The model
            inputs: Input batch
            num_items_in_batch: Number of items in batch (for newer transformers versions)
            
        Returns:
            Loss tensor
        """
        # Let the parent trainer handle the full training step,
        # our custom compute_loss method will be called automatically
        loss = super().training_step(model, inputs, num_items_in_batch)
        # Sanitize gradients after backward to avoid NaNs propagating
        if model is not None:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad = parameter.grad
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        grad.nan_to_num_(nan=0.0, posinf=1e3, neginf=-1e3)
            # Optional gradient clipping
            if hasattr(self.args, 'max_grad_norm') and self.args.max_grad_norm and self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        return loss
    
    def _on_epoch_begin_logic(self, args, state, control, model=None, **kwargs):
        """Internal logic for epoch begin processing - called by the callback."""
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {state.epoch}")
        
        # Verify watermark before training (only if enabled)
        if self.watermark_enabled and state.epoch == 0:
            verification_results = self.watermark_embedder.verify_watermark(model)
            logger.info(f"Initial watermark verification: {verification_results}")
    
    def _on_epoch_end_logic(self, args, state, control, model=None, **kwargs):
        """Internal logic for epoch end processing - called by the callback."""
        logger.info(f"INTERNAL: _on_epoch_end_logic called for epoch {state.epoch}")
        
        epoch_time = time.time() - (self.epoch_start_time or time.time())
        self.training_metrics['epoch_times'].append(epoch_time)
        
        logger.info(f"Epoch {state.epoch} completed in {epoch_time:.2f} seconds")
        
        # Verify watermark after each epoch (with timing)
        watermark_verification_time = 0.0
        if self.watermark_enabled:
            verification_start_time = time.time()
            verification_results = self.watermark_embedder.verify_watermark(model)
            watermark_verification_time = time.time() - verification_start_time
            current_match_rate = verification_results.get('overall_match_rate', 0.0)
            current_ber = verification_results.get('overall_ber', 1.0)
            logger.info(f"Watermark verification completed in {watermark_verification_time:.4f} seconds")
        else:
            verification_results = {}
            current_match_rate = -1.0
            current_ber = 1.0
        selected_metric_value = float(current_match_rate)
        selected_threshold = float(self.match_success_threshold)
        
        if self.watermark_enabled:
            self.training_metrics['watermark_verification'].append({
                'epoch': state.epoch,
                'verification': verification_results,
                'timestamp': time.time()
            })
        
        # Log watermark metrics
        # Evaluate language task if in dual-purpose training (with timing)
        language_loss = None
        language_eval_time = 0.0
        if self.is_dual_training and state.epoch % self.language_eval_frequency == 0:
            try:
                # Get eval dataloader if available
                eval_dataloader = self.get_eval_dataloader()
                language_eval_start_time = time.time()
                language_loss = self._evaluate_language_task(model, eval_dataloader)
                language_eval_time = time.time() - language_eval_start_time
                if language_loss is not None:
                    self.language_performance_history.append({
                        'epoch': state.epoch,
                        'language_loss': language_loss
                    })
                    logger.info(f"Language Task Loss: {language_loss:.4f} (target: {self.language_success_threshold:.4f})")
                    logger.info(f"Language task evaluation completed in {language_eval_time:.4f} seconds")
                    
                    # Check if language task is satisfied
                    if language_loss <= self.language_success_threshold:
                        self.language_task_completed = True
                        logger.info(f"Language task threshold reached! Loss: {language_loss:.4f} <= {self.language_success_threshold:.4f}")
            except Exception as e:
                logger.warning(f"Language evaluation failed: {e}")
        
        if self.watermark_enabled:
            logger.info(
                f"Epoch {state.epoch} Watermark Metrics: "
                f"match_rate={current_match_rate:.6f}, BER={current_ber:.6f} | "
                f"threshold={selected_threshold:.2f}"
            )

        # Record CSV row for this epoch
        try:
            current_lr = None
            if hasattr(self, 'optimizer') and self.optimizer is not None and len(self.optimizer.param_groups) > 0:
                current_lr = float(self.optimizer.param_groups[0].get('lr', None))
            self.csv_records.append({
                'epoch': float(state.epoch),
                'shard': int(self.current_shard),
                'learning_rate': current_lr,
                'watermark_rate': float(current_match_rate) if self.watermark_enabled else -1.0,
                'task_loss': float(language_loss) if language_loss is not None else None,
                'epoch_completed_timestamp': int(time.time()),
                'watermark_verification_time': float(watermark_verification_time),
                'language_eval_time': float(language_eval_time),
            })
        except Exception:
            pass
        
        # Dynamic stopping logic
        if state.epoch >= self.min_epochs:
            # Check if watermark threshold is reached (selected metric)
            if self.watermark_enabled and selected_metric_value >= selected_threshold:
                logger.info(f"SUCCESS! Watermark threshold reached at epoch {state.epoch}!")
                logger.info(
                    f"   match_rate: {selected_metric_value:.6f} >= {selected_threshold}"
                )
                
                # Handle shard completion for dual-purpose training
                if self.is_dual_training and self.shard_config.get('enabled', False):
                    self._complete_current_shard(state, selected_metric_value, control, model)
                    return
                else:
                    logger.info(f"STOPPING TRAINING: Setting control.should_training_stop = True")
                    control.should_training_stop = True
                    return
            
            # Check for improvement and early stopping
            if self.early_stopping_config.get('enabled', False):
                if selected_metric_value > self.best_metric_value:
                    self.best_metric_value = selected_metric_value
                    self.epochs_without_improvement = 0
                    logger.info(
                        f"New best watermark match_rate: {selected_metric_value:.6f}"
                    )
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"No improvement for {self.epochs_without_improvement} epochs")
                    
                    patience = self.early_stopping_config.get('patience', 3)
                    if self.epochs_without_improvement >= patience:
                        logger.info(f"Early stopping triggered after {self.epochs_without_improvement} epochs without improvement")
                        # In dual-purpose shard training, end shard and start a new one
                        if self.is_dual_training and self.shard_config.get('enabled', False):
                            logger.info("Ending current shard due to no improvement and starting a new shard")
                            self._complete_current_shard(state, selected_metric_value, control, model)
                            return
                        # Otherwise, stop the entire training run
                        control.should_training_stop = True
                        return
        else:
            logger.info(f"Minimum epochs not met ({state.epoch}/{self.min_epochs}), continuing training...")
            
        # Update best metric for bookkeeping (non-early-stop path)
        if self.watermark_enabled and (state.epoch == 1 or selected_metric_value > self.best_metric_value):
            self.best_metric_value = selected_metric_value
    
    def _complete_current_shard(self, state, watermark_success_rate, control, model):
        """Complete the current shard and decide whether to start a new one."""
        # Record shard completion
        current_language_loss = None
        if self.language_performance_history:
            current_language_loss = self.language_performance_history[-1].get('language_loss', 'N/A')
        
        shard_info = {
            'shard_number': self.current_shard,
            'start_epoch': getattr(self, 'shard_start_epoch', 1),
            'end_epoch': state.epoch,
            'epochs_duration': state.epoch - getattr(self, 'shard_start_epoch', 1) + 1,
            'watermark_success_rate': watermark_success_rate,
            'language_loss': current_language_loss,
            'training_loss': getattr(state, 'log_history', [{}])[-1].get('train_loss', 'N/A')
        }
        self.shard_history.append(shard_info)
        self.total_shards = self.current_shard
        
        logger.info(f"SHARD {self.current_shard} COMPLETED!")
        logger.info(f"   Duration: {shard_info['epochs_duration']} epochs (epoch {shard_info['start_epoch']} → {shard_info['end_epoch']})")
        logger.info(f"   Watermark success: {watermark_success_rate:.4f}")
        logger.info(f"   Language loss: {current_language_loss}")
        logger.info(f"   Training loss: {shard_info['training_loss']}")
        
        # Check if language task is completed (primary stopping condition)
        if self.language_task_completed:
            logger.info(f"TRAINING COMPLETE: Language task satisfied after {self.current_shard} shards!")
            logger.info(f"   Language loss threshold: {self.language_success_threshold:.2f}")
            logger.info(f"   Watermark embedding: Successfully embedded {self.current_shard} watermarks")
            control.should_training_stop = True
            return
        
        # Check if we've reached maximum shards (secondary stopping condition)
        max_shards = self.shard_config.get('max_shards', 5)
        if self.current_shard >= max_shards:
            logger.info(f"TRAINING COMPLETE: Reached maximum shards ({max_shards})")
            logger.info(f"   Language task not yet satisfied (current loss: {current_language_loss})")
            control.should_training_stop = True
            return
        
        # Continue to next shard
        logger.info(f"   Continuing to next shard (language task not satisfied)")
        self._start_new_shard(state, model)
    
    def _start_new_shard(self, state, model):
        """Start a new watermark embedding shard with completely fresh watermark."""
        self.current_shard += 1
        self.shard_start_epoch = state.epoch + 1
        
        # Reset watermark tracking for new shard
        self.best_metric_value = 0.0
        self.epochs_without_improvement = 0
        
        # Generate completely new watermark for the new shard
        logger.info(f"STARTING SHARD {self.current_shard}")
        logger.info(f"   Generating completely new watermark based on current model weights...")
        
        # Store old watermark for comparison
        old_watermark = self.watermark_embedder.watermark_bits.clone() if self.watermark_enabled else None
        
        # Generate new watermark using model weights hash as seed
        # This ensures each shard has a unique watermark based on the 
        # model state at the end of the previous shard
        if self.watermark_enabled:
            new_watermark = self.watermark_embedder._generate_watermark(model)
        else:
            new_watermark = None
        
        # Log the change
        if self.watermark_enabled:
            logger.info(f"   Old watermark: {old_watermark[:8]}... → New watermark: {new_watermark[:8]}...")
            logger.info(f"   Watermark verification will now check against NEW bits")
        
        # Note: New embedding positions will be generated automatically when
        # embed_watermark_in_layer is called for each layer during training
        
        if self.watermark_enabled:
            logger.info(f"   New {self.watermark_embedder.bit_length}-bit watermark generated for shard {self.current_shard}")
            logger.info(f"   Fresh embedding positions will be selected per layer during training")
            logger.info(f"   Watermark uniqueness guaranteed by model checkpoint hash")
        
        # Verify the match rate should reset with new watermark
        if self.watermark_enabled:
            initial_verification = self.watermark_embedder.verify_watermark(model)
            initial_match_rate = initial_verification.get('overall_match_rate', 0.0)
            logger.info(f"   Initial match rate with new watermark: {initial_match_rate:.6f} (should be low ~0.5)")
            
            if initial_match_rate > 0.7:
                logger.warning(f"   Match rate unexpectedly high! New watermark may not be properly applied.")

        # Also log the initial language loss at shard start so we don't miss it
        try:
            eval_dataloader = self.get_eval_dataloader()
            initial_lang_loss = self._evaluate_language_task(model, eval_dataloader)
            if initial_lang_loss is not None:
                self.language_performance_history.append({
                    'epoch': state.epoch,  # shard start moment
                    'language_loss': initial_lang_loss,
                    'event': 'shard_start'
                })
                logger.info(f"   Initial language loss at shard start: {initial_lang_loss:.4f} (target: {self.language_success_threshold:.2f})")
                # NOTE: We don't add a CSV record here - let the training naturally progress to the next epoch
        except Exception as e:
            logger.warning(f"   Failed to compute initial language loss at shard start: {e}")
    
    def _evaluate_language_task(self, model, eval_dataloader=None):
        """Evaluate language modeling performance on validation data."""
        if not self.is_dual_training or eval_dataloader is None:
            return None
        
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                
                # Compute only base language modeling loss (not watermark loss)
                outputs = model(**batch)
                logits = outputs.logits
                labels = batch['labels']
                
                # Standard language modeling loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit evaluation to avoid long delays
                if num_batches >= 20:  # Evaluate on max 20 batches
                    break
        
        model.train()
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        """Called when logging."""
        if logs and hasattr(self, '_current_loss_components'):
            # Add loss components to logs
            logs.update(self._current_loss_components)
            
            # Store detailed loss components
            self.training_metrics['loss_components'].append({
                'step': state.global_step,
                'epoch': state.epoch,
                **self._current_loss_components
            })
    
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval"
    ) -> Dict[str, float]:
        """
        Evaluate the model with custom metrics.
        
        Args:
            eval_dataset: Evaluation dataset
            ignore_keys: Keys to ignore in evaluation
            metric_key_prefix: Prefix for metric keys
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Standard evaluation
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Add perplexity (if eval_loss available)
        try:
            eval_loss_val = float(eval_results.get(f"{metric_key_prefix}_loss", eval_results.get("eval_loss")))
            if eval_loss_val is not None and np.isfinite(eval_loss_val):
                eval_results[f"{metric_key_prefix}_perplexity"] = float(np.exp(eval_loss_val))
        except Exception:
            pass
        
        # Add simple next-token accuracy over a few batches
        try:
            dataloader = self.get_eval_dataloader(eval_dataset)
            model = self.model
            model.eval()
            correct, total = 0, 0
            batches = 0
            with torch.no_grad():
                for batch in dataloader:
                    batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
                    outputs = model(**batch)
                    logits = outputs.logits
                    labels = batch['labels']
                    # Shift for causal LM
                    preds = logits.argmax(dim=-1)
                    pred_shift = preds[..., :-1].contiguous()
                    label_shift = labels[..., 1:].contiguous()
                    mask = label_shift.ne(-100) if (-100 in label_shift) else torch.ones_like(label_shift, dtype=torch.bool)
                    matches = (pred_shift == label_shift) & mask
                    correct += matches.sum().item()
                    total += mask.sum().item()
                    batches += 1
                    if batches >= 10:
                        break
            if total > 0:
                eval_results[f"{metric_key_prefix}_token_accuracy"] = correct / total
        except Exception:
            pass
        
        # Add watermark verification to evaluation (only if watermark is enabled)
        if self.watermark_enabled:
            verification_results = self.watermark_embedder.verify_watermark(self.model)
            
            # Add watermark metrics to evaluation results
            eval_results[f"{metric_key_prefix}_watermark_match_rate"] = verification_results['overall_match_rate']
            eval_results[f"{metric_key_prefix}_watermark_ber"] = verification_results['overall_ber']
            eval_results[f"{metric_key_prefix}_watermark_success"] = float(verification_results['success'])
            
            # Add per-layer match rates
            match_rates_per_layer = verification_results.get('match_rate_per_layer', {})
            for layer_name, match_rate in match_rates_per_layer.items():
                safe_layer_name = layer_name.replace('.', '_')
                eval_results[f"{metric_key_prefix}_watermark_{safe_layer_name}_match_rate"] = match_rate
        else:
            # Add placeholder metrics when watermark is disabled
            eval_results[f"{metric_key_prefix}_watermark_match_rate"] = -1.0
            eval_results[f"{metric_key_prefix}_watermark_ber"] = -1.0
            eval_results[f"{metric_key_prefix}_watermark_success"] = -1.0
        
        return eval_results

    def save_epoch_csv(self, out_dir: Any) -> Optional[str]:
        """Write collected epoch-level metrics to a CSV file in output dir.

        Returns the path of the written CSV file.
        """
        try:
            from pathlib import Path
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"epoch_metrics_{timestamp}.csv"
            csv_path = out_path / filename

            fieldnames = ['epoch', 'shard', 'learning_rate', 'watermark_rate', 'task_loss', 'epoch_completed_timestamp', 'watermark_verification_time', 'language_eval_time']
            with csv_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.csv_records:
                    writer.writerow(row)
            logger.info(f"Epoch metrics CSV written to: {csv_path}")
            return str(csv_path)
        except Exception as e:
            logger.warning(f"Failed to write epoch metrics CSV: {e}")
            return None
    
    def get_train_dataloader(self):
        """Get training dataloader with custom collation if needed."""
        return super().get_train_dataloader()
    
    def save_training_metrics(self, output_dir: str):
        """Save detailed training metrics to file."""
        metrics_path = Path(output_dir) / "training_metrics.json"
        
        # Add final statistics
        self.training_metrics['total_training_time'] = sum(self.training_metrics['epoch_times'])
        self.training_metrics['config'] = self.config
        if self.watermark_enabled:
            self.training_metrics['watermark_info'] = self.watermark_embedder.get_watermark_info()
        else:
            self.training_metrics['watermark_info'] = {'status': 'watermark_disabled'}
        
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=2, default=str)
        
        logger.info(f"Training metrics saved to {metrics_path}")
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model with optional watermark information."""
        super().save_model(output_dir, _internal_call)
        
        if output_dir:
            # Save watermark embedder (only if watermark is enabled)
            if self.watermark_enabled:
                watermark_path = Path(output_dir) / "watermark.pt"
                self.watermark_embedder.save_watermark(str(watermark_path))
                logger.info(f"Watermark metadata saved to: {watermark_path}")
            else:
                logger.info("Watermark disabled - no watermark metadata saved")
            
            # Save training metrics
            self.save_training_metrics(output_dir)
            
            # Save final verification results (only if watermark is enabled)
            if self.watermark_enabled:
                verification_results = self.watermark_embedder.verify_watermark(self.model)
                verification_path = Path(output_dir) / "final_verification.json"
                with open(verification_path, 'w') as f:
                    json.dump(verification_results, f, indent=2, default=str)
    
    def get_watermark_stats(self) -> Dict[str, Any]:
        """Get comprehensive watermark statistics."""
        if not self.watermark_enabled:
            return {
                'watermark_enabled': False,
                'success_achieved': False,
                'final_match_rate': -1.0,
                'final_ber': -1.0
            }
            
        verification_results = self.watermark_embedder.verify_watermark(self.model)
        
        stats = {
            'current_verification': verification_results,
            'watermark_info': self.watermark_embedder.get_watermark_info(),
            'verification_history': self.training_metrics['watermark_verification'],
            'success_achieved': verification_results['success'],
            'final_match_rate': verification_results['overall_match_rate'],
            'final_ber': verification_results['overall_ber'],
            # Shard information for dual-purpose training
            'shard_info': {
                'is_dual_training': self.is_dual_training,
                'current_shard': self.current_shard,
                'total_shards_completed': self.total_shards,
                'shard_history': self.shard_history,
                'shard_config': self.shard_config,
                'language_performance_history': self.language_performance_history,
                'language_task_completed': self.language_task_completed,
                'language_success_threshold': self.language_success_threshold
            }
        }
        
        # Calculate success epoch if achieved
        for record in self.training_metrics['watermark_verification']:
            if record['verification']['success']:
                stats['success_epoch'] = record['epoch']
                break
        
        return stats


class WatermarkTrainingArguments(TrainingArguments):
    """Extended training arguments with watermark-specific parameters."""
    
    def __init__(self, *args, **kwargs):
        # Extract watermark-specific arguments
        self.watermark_verification_steps = kwargs.pop('watermark_verification_steps', 100)
        self.early_stop_on_watermark = kwargs.pop('early_stop_on_watermark', False)
        
        super().__init__(*args, **kwargs)


def create_watermark_trainer(
    model: nn.Module,
    tokenizer,
    train_dataset,
    eval_dataset,
    config: Dict[str, Any],
    watermark_embedder: WatermarkEmbedder
) -> WatermarkTrainer:
    """
    Factory function to create a watermark trainer.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        config: Configuration dictionary
        watermark_embedder: Watermark embedder instance
        
    Returns:
        Configured WatermarkTrainer instance
    """
    training_config = config['training']
    
    # Create training arguments
    training_args = WatermarkTrainingArguments(
        output_dir=config['output']['model_dir'],
        num_train_epochs=training_config.get('max_epochs', training_config.get('epochs', 3)),
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=config['evaluation']['eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        eval_strategy=training_config['evaluation_strategy'],
        save_strategy=training_config['save_strategy'],
        load_best_model_at_end=False,  # Disabled when eval_strategy is "no"
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb by default
        dataloader_num_workers=0,  # Disable multiprocessing to avoid tokenizer issues
        fp16=config['hardware']['mixed_precision'] if torch.cuda.is_available() else False,
    )
    
    # Create data collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Optionally freeze non-target layers to focus updates on watermark positions
    # In dual-purpose training, keep all layers trainable so language loss can improve
    dual_training = config['loss']['base_loss_weight'] > 0.0
    if not dual_training:
        target_patterns = watermark_embedder.watermark_config.get('target_layers', [])
        def _is_target(name: str) -> bool:
            for pattern in target_patterns:
                if pattern in name:
                    return True
            return False

        for name, param in model.named_parameters():
            # keep trainable only for target patterns; freeze others
            if not _is_target(name):
                param.requires_grad = False

    # Create trainer
    trainer = WatermarkTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        watermark_embedder=watermark_embedder,
        config=config,
    )
    
    # Register the watermark early stopping callback
    watermark_callback = WatermarkEarlyStoppingCallback(trainer)
    trainer.add_callback(watermark_callback)
    logger.info("Registered WatermarkEarlyStoppingCallback")
    
    return trainer
