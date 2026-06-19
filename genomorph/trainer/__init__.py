from .grpo_config import DNALLMGRPOConfig

def __getattr__(name):
    if name == "DNALLMGRPOTrainer":
        from .grpo_trainer import DNALLMGRPOTrainer
        return DNALLMGRPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DNALLMGRPOConfig",
    "DNALLMGRPOTrainer",
]