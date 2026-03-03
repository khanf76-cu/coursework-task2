# src/ppo.py
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import torch.nn as nn

@dataclass
class Trajectory:
    obs: List[np.ndarray]
    masks: List[np.ndarray]
    actions: List[int]
    logprobs: List[float]
    values: List[float]
    rewards: List[float]
    dones: List[bool]

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - float(dones[t])
        next_value = values[t + 1] if (t + 1) < T else 0.0
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns

def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    traj: Trajectory,
    device: str,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    batch_size: int,
):
    obs = torch.tensor(np.array(traj.obs), device=device)
    masks = torch.tensor(np.array(traj.masks), device=device)
    actions = torch.tensor(np.array(traj.actions), device=device)
    old_logprobs = torch.tensor(np.array(traj.logprobs), device=device)
    values_np = np.array(traj.values, dtype=np.float32)

    rewards = np.array(traj.rewards, dtype=np.float32)
    dones = np.array(traj.dones, dtype=np.bool_)

    adv, rets = compute_gae(rewards, values_np, dones, gamma=gamma, lam=gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    adv_t = torch.tensor(adv, device=device)
    rets_t = torch.tensor(rets, device=device)

    N = obs.shape[0]
    idxs = np.arange(N)

    neg_inf = torch.tensor(-1e9, device=device)

    for _ in range(ppo_epochs):
        np.random.shuffle(idxs)
        for start in range(0, N, batch_size):
            mb = idxs[start : start + batch_size]

            logits, v = model(obs[mb])
            masked_logits = torch.where(masks[mb] > 0.0, logits, neg_inf)
            dist = torch.distributions.Categorical(logits=masked_logits)

            new_logp = dist.log_prob(actions[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logprobs[mb])
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((v - rets_t[mb]) ** 2).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()