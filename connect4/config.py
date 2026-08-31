from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch_size: int = 10
    eps_start: float = 0.9
    eps_end: float = 0.1
    gamma: float = 0.99
    eps_decay: int = 10000
    tau: float = 0.002
    lr: float = 0.001
    memory_size: int = 10_000
    episodes: int = 300
    checkpoint_interval: int = 50
    save_dir: str = "checkpoints"
