


**Coursework Task 2 — Chef’s Hat Gym Reinforcement Learning Agent**  
**Student ID mod 7 = 3 — Sparse / Delayed Reward Variant**

---

## 1. Overview

This project implements a reinforcement learning agent for the **Chef’s Hat Gym** multi-agent card game environment.

Chef’s Hat is a competitive, turn-based, multi-agent environment with:

- Large discrete action space
- Sparse and delayed rewards (match outcome at the end)
- Non-stationary opponents
- Multi-agent interactions

For this coursework, the assigned variant is:

> **ID mod 7 = 3 — Sparse / Delayed Reward Variant**

The focus is on handling delayed terminal rewards through **reward shaping techniques** and analysing their impact on learning behaviour.

---

## 2. Environment

Official environment used:

- Chef’s Hat Gym
- Installed via `pip install chefshatgym`
- Gym version: `gym==0.26.2`
- Python 3.10 (Anaconda environment)

The agent is trained in local game rooms using:

```python
ChefsHatRoomLocal
````

---

## 3. State Representation

The environment provides a fixed-length observation vector (size ≈ 228).

The agent directly uses this observation vector as input to a neural network.

No handcrafted feature engineering was applied. The neural network learns representations end-to-end.

---

## 4. Action Handling Strategy

Chef’s Hat has a large discrete action space (~200 actions).

The PPO policy outputs logits for all possible actions. The selected action is sampled from the policy distribution.

Invalid actions are automatically handled by the environment, but:

* We detect when an action becomes random (`Action_Random`)
* We penalise invalid actions via reward shaping

This encourages valid strategic behaviour.

---

## 5. Reward Design (Variant mod 3 Focus)

Chef’s Hat provides **sparse terminal rewards** at the end of matches.

To improve credit assignment, we apply reward shaping:

### 5.1 Terminal Reward (Sparse)

At match end:

```
Match_Score[player0]
```

This is the true game objective.

### 5.2 Invalid Action Penalty

If the environment replaces an action (invalid move):

```
Action_Random != ""
```

We apply:

```
-0.05 penalty
```

This discourages illegal behaviour.

### 5.3 Card Progress Shaping (Dense Reward)

We reward reduction in remaining cards:

```
shaping_reward = scale * (previous_cards - current_cards)
```

This encourages faster completion of rounds.

### Shaping Strength Experiments

We evaluate multiple shaping strengths:

* 0.01
* 0.02
* 0.05

To analyse the trade-off between sparse and dense rewards.

---

## 6. RL Algorithm

We implement:

> **Proximal Policy Optimisation (PPO)**
> Actor–Critic architecture

Key parameters:

| Parameter     | Value |
| ------------- | ----- |
| Learning Rate | 3e-4  |
| Gamma         | 0.99  |
| GAE Lambda    | 0.95  |
| Clip Epsilon  | 0.2   |
| Entropy Coef  | 0.01  |
| Episodes      | 300   |

---

## 7. Evaluation Metrics

Performance is evaluated using:

1. **Win Rate**
   Percentage of games where the agent finishes first.

2. **Average Game Score**
   Based on `Game_Score[player0]`.

3. **Average Performance Score**
   Environment-provided metric: `Game_Performance_Score[player0]`.

4. **Learning Curves**

   * Performance vs Episode
   * Score vs Episode
   * Episode length vs Episode

5. **Reward Breakdown**

   * Terminal reward
   * Shaping reward

6. **Invalid Action Rate**

   * Shows improvement in valid decision-making.

7. **Policy Entropy**

   * Exploration → Exploitation transition

---

## 8. Experiments

### 8.1 Shaping ON vs OFF

We compare:

| Config | Win Rate | Avg Perf           |
| ------ | -------- | ------------------ |
| OFF    | lower    | slower convergence |
| ON     | higher   | faster convergence |

Reward shaping improves:

* Convergence speed
* Stability
* Final performance

---

### 8.2 Shaping Strength Sweep

| Scale | Behaviour          |
| ----- | ------------------ |
| 0.01  | Slow but stable    |
| 0.02  | Best performance   |
| 0.05  | Slight instability |

Moderate shaping achieves best balance.

---

## 9. Results

Example final evaluation (50 games):

```
Win Rate: 0.80
Average Score: 7.2
Average Performance Score: 1.13
```

Learning curves and analysis plots are stored in:

```
results/plots/
```

Summary tables are stored in:

```
results/summary_tables/
```

---

## 10. Repository Structure

```
coursework-task2/
│
├── src/
│   ├── config.py
│   ├── model.py
│   ├── ppo.py
│   ├── env_runner.py
│   ├── train.py
│   ├── analysis.py
│
├── models/
├── results/
│   ├── plots/
│   └── summary_tables/
│
└── README.md
```

---

## 11. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train agent

```bash
python -m src.train
```

### Generate analysis plots

```bash
python -m src.analysis
```

---

## 12. Limitations

* Opponents are random agents (no opponent modelling).
* Shaping may bias behaviour toward card reduction rather than optimal long-term strategy.
* PPO hyperparameters not exhaustively tuned.
* Computational constraints limited large-scale experiments.

---

## 13. Conclusion

This project demonstrates:

* Effective handling of sparse rewards using shaping
* Improved learning stability and convergence
* Strong competitive performance against random agents

Reward shaping significantly improves training efficiency in delayed-reward multi-agent environments such as Chef’s Hat Gym.

