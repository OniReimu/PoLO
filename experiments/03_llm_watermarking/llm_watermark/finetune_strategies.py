"""
Finetuning strategy utilities (standard vs LoRA), compatible with watermarking.
"""

from typing import Dict, Tuple
import logging


logger = logging.getLogger(__name__)


def apply_finetune_strategy(model, config: Dict) -> Tuple[object, Dict]:
    """Apply finetuning strategy to the model (standard or LoRA).

    Returns the (possibly wrapped) model and a dict with strategy info.
    """
    finetune_cfg = config.get('finetune', {}) or {}
    strategy = finetune_cfg.get('strategy', 'standard')

    if strategy == 'lora':
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as e:
            logger.error(f"LoRA selected but peft is unavailable: {e}")
            raise

        lora_cfg = finetune_cfg.get('lora', {}) or {}
        # Use watermark target layers as default LoRA target modules
        default_targets = config.get('watermark', {}).get('target_layers', [])
        target_modules = lora_cfg.get('target_modules', default_targets)
        if isinstance(target_modules, str):
            target_modules = [m.strip() for m in target_modules.split(',') if m.strip()]

        r = int(lora_cfg.get('r', 8))
        alpha = int(lora_cfg.get('alpha', 16))
        dropout = float(lora_cfg.get('dropout', 0.05))
        bias = lora_cfg.get('bias', 'none')

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=bias,
        )

        logger.info(
            f"Applying LoRA: r={r}, alpha={alpha}, dropout={dropout}, targets={target_modules}"
        )

        model = get_peft_model(model, lora_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

        info = {
            'strategy': 'lora',
            'target_modules': target_modules,
            'r': r,
            'alpha': alpha,
            'dropout': dropout,
        }
        return model, info

    # Default: standard full/partial finetuning (no wrapping)
    logger.info("Using standard fine-tuning (no adapters)")
    return model, {'strategy': 'standard'}


