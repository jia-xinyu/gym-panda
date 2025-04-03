import gymnasium as gym
import numpy as np

import gym_panda


def test_seed_pick_and_place():
    final_observations = []
    env = gym.make("gym_panda/PandaPickAndPlace-v3")
    actions = [
        np.array([0.429, -0.287, 0.804, -0.592]),
        np.array([0.351, -0.136, 0.296, -0.223]),
        np.array([-0.187, 0.706, -0.988, 0.972]),
        np.array([-0.389, -0.249, 0.374, -0.389]),
        np.array([-0.191, -0.297, -0.739, 0.633]),
        np.array([0.093, 0.242, -0.11, -0.949]),
    ]
    for _ in range(2):
        env.reset(seed=794512)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)

    assert np.allclose(final_observations[0]["agent_pos"], final_observations[1]["agent_pos"])
    assert np.allclose(final_observations[0]["object_pos"], final_observations[1]["object_pos"])


def test_seed_stack():
    final_observations = []
    env = gym.make("gym_panda/PandaPickAndPlace-v3")
    actions = [
        np.array([-0.609, 0.73, -0.433, 0.76]),
        np.array([0.414, 0.327, 0.275, -0.196]),
        np.array([-0.3, 0.589, -0.712, 0.683]),
        np.array([0.772, 0.333, -0.537, -0.253]),
        np.array([0.784, -0.014, -0.997, -0.118]),
        np.array([-0.12, -0.958, -0.744, -0.98]),
    ]
    for _ in range(2):
        env.reset(seed=657894)
        for action in actions:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset()
        final_observations.append(observation)
    assert np.allclose(final_observations[0]["agent_pos"], final_observations[1]["agent_pos"])
    assert np.allclose(final_observations[0]["object_pos"], final_observations[1]["object_pos"])
