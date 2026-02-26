"""
GEKO LoRA/PEFT Integration

Provides one-liner LoRA wrapping for any HuggingFace model.
Requires: pip install peft  (or pip install gekolib[peft])

Usage:
    from peft import LoraConfig
    from geko import GEKOTrainer

    trainer = GEKOTrainer(
        model=model,
        train_dataset=dataset,
        lora_config=LoraConfig(r=16, target_modules=["q_proj", "v_proj"]),
    )
"""


def is_peft_available() -> bool:
    """Check if the peft library is installed."""
    try:
        import peft  # noqa: F401
        return True
    except ImportError:
        return False


def apply_lora(model, lora_config):
    """
    Wrap a model with LoRA adapters using PEFT.

    Args:
        model: Any HuggingFace model
        lora_config: A peft.LoraConfig instance

    Returns:
        The model wrapped with LoRA adapters (trainable params drastically reduced)

    Raises:
        ImportError: If peft is not installed
    """
    try:
        from peft import get_peft_model
    except ImportError:
        raise ImportError(
            "peft is required for LoRA integration. Install it with:\n"
            "    pip install peft\n"
            "or:\n"
            "    pip install gekolib[peft]"
        )

    model = get_peft_model(model, lora_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"[GEKO] LoRA applied\n"
        f"  Trainable params : {trainable_params:,} ({trainable_params / total_params:.2%})\n"
        f"  Total params     : {total_params:,}\n"
        f"  Frozen params    : {total_params - trainable_params:,}"
    )

    return model
