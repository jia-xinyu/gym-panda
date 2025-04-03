import os
from gymnasium.envs.registration import register

with open(os.path.join(os.path.dirname(__file__), "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

ENV_IDS = []

for task in ["PickAndPlace", "Stack"]:
    for control_type in ["ee", "joints"]:
        control_suffix = "Joints" if control_type == "joints" else ""
        env_id = f"gym_panda/Panda{task}{control_suffix}-v3"

        register(
            id=env_id,
            entry_point=f"gym_panda.panda_tasks:PandaEnv",
            # Even after seeding, the rendered observations are slightly different,
            # so we set `nondeterministic=True` to pass `check_env` tests
            nondeterministic=True,
            kwargs={"obs_type": "state", "task": task, "control_type": control_type},
            max_episode_steps=100 if task == "Stack" else 50,
        )

        ENV_IDS.append(env_id)
