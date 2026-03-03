# src/train.py
import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from .config import Config
from .model import ActorCritic
from .ppo import ppo_update
from .env_runner import run_one_episode
from .eval import evaluate, extract_metrics

def main():
    cfg = Config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Reproducibility
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    model = ActorCritic().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Logging
    train_my_score = []
    train_my_perf = []
    episode_steps = []

    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    for ep in range(cfg.total_episodes):
        traj, summary = run_one_episode(model, device, cfg, seed=cfg.seed + ep)

        # PPO update
        ppo_update(
            model=model,
            optimizer=optimizer,
            traj=traj,
            device=device,
            clip_eps=cfg.clip_eps,
            entropy_coef=cfg.entropy_coef,
            value_coef=cfg.value_coef,
            max_grad_norm=cfg.max_grad_norm,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            ppo_epochs=cfg.ppo_epochs,
            batch_size=cfg.batch_size,
        )

        my_score, my_perf, _ = extract_metrics(summary, player_index=0)
        train_my_score.append(my_score if my_score is not None else np.nan)
        train_my_perf.append(my_perf if my_perf is not None else np.nan)
        episode_steps.append(len(traj.rewards))

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{cfg.total_episodes} | steps={episode_steps[-1]} | score={train_my_score[-1]} | perf={train_my_perf[-1]}")

        if (ep + 1) % 50 == 0:
            eval_res = evaluate(model, device, cfg, n_games=20, seed=2000 + ep)
            print("Eval (20 games):", eval_res)

    # Save model
    save_path = os.path.join("models", "ppo_variant3.pt")
    torch.save(model.state_dict(), save_path)
    print("Saved model to:", save_path)

    # Plots
    plt.figure()
    plt.plot(train_my_perf)
    plt.title("Training: Game_Performance_Score (Player 0)")
    plt.xlabel("Episode")
    plt.ylabel("Perf Score")
    plt.savefig("results/perf_curve.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(train_my_score)
    plt.title("Training: Game_Score (Player 0)")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig("results/score_curve.png", dpi=200)
    plt.show()

    plt.figure()
    plt.plot(episode_steps)
    plt.title("Episode length (steps)")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig("results/steps_curve.png", dpi=200)
    plt.show()

    # Final evaluation
    final_eval = evaluate(model, device, cfg, n_games=50, seed=9999)
    print("Final evaluation (50 games):", final_eval)

if __name__ == "__main__":
    main()