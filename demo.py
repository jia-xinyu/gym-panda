import gymnasium as gym
import gym_panda, time
import numpy as np

env = gym.make("gym_panda/PandaPickAndPlace-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random action
    # action[3:] = np.array([0., 0., 0., 0.]) # fixed orientation
    # action[:3] = np.array([0., 0., 0.,])    # fixed position
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
    time.sleep(0.1)

time.sleep(8)
env.close()