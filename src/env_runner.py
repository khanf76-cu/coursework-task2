# src/env_runner.py
from typing import Dict, Any, Tuple, List
import numpy as np
import torch

from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon
from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer

from .model import masked_categorical
from .ppo import Trajectory
from .config import Config

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class RLPlayerPPO(ChefsHatPlayer):
    """
    Uses observation len 228, where last 200 entries are action mask.
    Returns one-hot action length 200.
    Collects trajectory for PPO.
    """
    def __init__(self, name, log_directory, model, device: str, cfg: Config, verbose_log=False):
        agent_suffix = "_RL"
        super().__init__(name=name, agent_suffix=agent_suffix, log_directory=log_directory, verbose_log=verbose_log)
        self.my_idx = 0
        self.model = model
        self.device = device
        self.cfg = cfg

        self.traj: Trajectory | None = None
        self._last_cards = None

        # filled after game:
        self.last_game_summary: Dict[str, Any] | None = None

    def reset_storage(self):
        self.traj = Trajectory([], [], [], [], [], [], [])
        self._last_cards = None
        self.last_game_summary = None

    def update_start_match(self, cards: list[float], players: list[str], starting_player: int):
        # Count non-zero cards as "cards in hand"
        self._last_cards = int(np.count_nonzero(np.array(cards)))

    def get_action(self, observation: list[float]):
        obs = np.array(observation, dtype=np.float32)
        action_mask = obs[28:228].copy()  # 200 mask entries

        obs_t = torch.tensor(obs, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(action_mask, device=self.device)

        with torch.no_grad():
            logits, value = self.model(obs_t)
            dist = masked_categorical(logits.squeeze(0), mask_t)
            action = dist.sample()
            logp = dist.log_prob(action)

        a = int(action.item())

        # Store transition
        assert self.traj is not None
        self.traj.obs.append(obs)
        self.traj.masks.append(action_mask)
        self.traj.actions.append(a)
        self.traj.logprobs.append(float(logp.item()))
        self.traj.values.append(float(value.item()))
        self.traj.rewards.append(0.0)
        self.traj.dones.append(False)

        # Return one-hot action
        one_hot = [0] * 200
        one_hot[a] = 1
        return one_hot

    def update_my_action(self, info: Dict[str, Any]):
        # Reward shaping (dense signals)
        if not self.cfg.use_reward_shaping:
            return
        if self.traj is None or len(self.traj.rewards) == 0:
            return

        shaping = 0.0

        # Invalid action penalty: if action ended up random/invalid
        # In your summary keys, Action_Random exists; sometimes it's '' or bool-like.
        action_random = info.get("Action_Random", "")
        if action_random is True:
            shaping += self.cfg.shaping_invalid_action_penalty
        elif isinstance(action_random, str) and action_random != "":
            # some versions store details as string; treat non-empty as "random happened"
            shaping += self.cfg.shaping_invalid_action_penalty

        # Card progress shaping
        cards_per_player = info.get("Cards_Per_Player", None)
        if isinstance(cards_per_player, list) and len(cards_per_player) > self.my_idx:
            my_cards = int(cards_per_player[self.my_idx])
            if self._last_cards is not None:
                delta = self._last_cards - my_cards
                shaping += self.cfg.shaping_card_progress_scale * float(delta)
            self._last_cards = my_cards

        self.traj.rewards[-1] += shaping

    def get_reward(self, info: Dict[str, Any]):
        """
        Sparse terminal reward: use Match_Score for this match.
        We add it to last timestep reward and mark done.
        """
        match_score = info.get("Match_Score", None)
        my_points = 0.0
        if isinstance(match_score, list) and len(match_score) > self.my_idx:
            my_points = float(match_score[self.my_idx])

        if self.traj is not None and len(self.traj.rewards) > 0:
            self.traj.rewards[-1] += my_points
            self.traj.dones[-1] = True

        return my_points

    def _choose_cards_to_give(self, cards: list[float], amount: int):
        # Always return a list of 'amount' cards (smallest non-zero)
        c = [x for x in cards if x != 0]
        c.sort()
        return c[:amount]

    # Common spelling in some versions (correct)
    def get_exchanged_cards(self, cards: list[float], amount: int):
        return self._choose_cards_to_give(cards, amount)
    
    # Misspelling used in other versions
    def get_exhanged_cards(self, cards: list[float], amount: int):
        return self._choose_cards_to_give(cards, amount)
    
    # Some versions call this
    def get_exchange_cards(self, cards: list[float], amount: int):
        return self._choose_cards_to_give(cards, amount)

def run_one_episode(
    model,
    device: str,
    cfg: Config,
    seed: int,
    opponents: int = 3,
) -> Tuple[Trajectory, Dict[str, Any]]:
    """
    Runs one game episode (STOP_CRITERIA_MATCHES matches).
    Player 0 is RL agent, others are random.
    Returns trajectory + last game summary (dict).
    """
    set_seeds(seed)

    room = ChefsHatRoomLocal(
        f"train_room_{seed}",
        game_type=ChefsHatEnv.GAMETYPE["MATCHES"],
        stop_criteria=cfg.stop_criteria_matches,
        max_rounds=-1,
        verbose_console=True,
        verbose_log=True,
        game_verbose_console=True,
        game_verbose_log=True,
        save_dataset=True,
    )

    log_dir = room.get_log_directory()

    rl = RLPlayerPPO("RL", log_dir, model=model, device=device, cfg=cfg, verbose_log=False)
    rl.reset_storage()
    room.add_player(rl)

    for i in range(opponents):
        room.add_player(AgentRandon(f"R{i}", log_dir, verbose_log=True))

    summary = room.start_new_game()
    rl.last_game_summary = summary

    return rl.traj, summary