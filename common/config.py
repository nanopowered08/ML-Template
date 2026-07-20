from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    max_context: int

@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    warmup_steps: int
    weight_decay: float

@dataclass
class TokenizerConfig:
    special_tokens: list[str]