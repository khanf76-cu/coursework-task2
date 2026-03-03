# src/eval.py
from typing import Dict, Any
import numpy as np
import torch

from .env_runner import run_one_episode
from .config import Config

def extract_metrics(summary: Dict[str, Any], player_index: int = 0):
    # Your version uses these keys:
    game_score = summary.get("Game_Score", None)
    perf = summary.get("Game_Performance_Score", None)

    my_score = None
    my_perf = None

    if isinstance(game_score, list) and len(game_score) > player_index:
        my_score = float(game_score[player_index])
    if isinstance(perf, list) and len(perf) > player_index:
        my_perf = float(perf[player_index])

    return my_score, my_perf, game_score

def evaluate(model, device: str, cfg: Config, n_games: int = 50, seed: int = 999) -> Dict[str, float]:
    model.eval()
    scores = []
    perfs = []
    wins = 0

    with torch.no_grad():
        for i in range(n_games):
            traj, summary = run_one_episode(model, device, cfg, seed=seed + i)
            my_score, my_perf, all_scores = extract_metrics(summary, player_index=0)

            if my_score is not None:
                scores.append(my_score)
            if my_perf is not None:
                perfs.append(my_perf)

            if isinstance(all_scores, list) and len(all_scores) == 4 and my_score is not None:
                if my_score == max(all_scores):
                    wins += 1

    model.train()

    return {
        "win_rate": wins / n_games,
        "avg_score": float(np.mean(scores)) if scores else float("nan"),
        "avg_perf": float(np.mean(perfs)) if perfs else float("nan"),
    }