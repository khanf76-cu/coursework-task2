# src/config.py
from dataclasses import dataclass

@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Training
    total_episodes: int = 300
    stop_criteria_matches: int = 3
    lr: float = 3e-4

    # PPO / GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 256

    # Variant mod 3: sparse reward + shaping
    use_reward_shaping: bool = True
    shaping_invalid_action_penalty: float = -0.05
    shaping_card_progress_scale: float = 0.02  # strong + safe