# src/model.py
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 228, act_dim: int = 200, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy = nn.Linear(hidden, act_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        x = self.shared(obs)
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        return logits, value


def masked_categorical(logits: torch.Tensor, action_mask: torch.Tensor):
    """
    logits: [act_dim]
    action_mask: [act_dim] (1 valid, 0 invalid)
    """
    neg_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(action_mask > 0.0, logits, neg_inf)
    dist = torch.distributions.Categorical(logits=masked_logits)
    return dist