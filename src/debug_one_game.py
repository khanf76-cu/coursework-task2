import os
from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.env import ChefsHatEnv
from ChefsHatGym.agents.agent_random import AgentRandon

def main():
    room = ChefsHatRoomLocal(
        "debug_room",
        game_type=ChefsHatEnv.GAMETYPE["MATCHES"],
        stop_criteria=1,
        max_rounds=-1,
        verbose_console=True,
        verbose_log=True,
        game_verbose_console=True,
        game_verbose_log=True,
        save_dataset=False,
    )

    log_dir = room.get_log_directory()
    print("Room log directory:", log_dir)

    # IMPORTANT: create per-agent folders
    for i in range(4):
        agent_name = f"R{i}"
        agent_folder = os.path.join(os.getcwd(), f"RANDOM_{agent_name}")
        os.makedirs(agent_folder, exist_ok=True)
        room.add_player(AgentRandon(agent_name, agent_folder, verbose_log=True))

    summary = room.start_new_game()
    print("Finished.")
    print("Temp folder:", os.path.abspath("temp"))

if __name__ == "__main__":
    main()