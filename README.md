# gym-panda

## Installation

```bash
git clone https://github.com/jia-xinyu/gym-panda.git
pip install -e gym-panda
```

## Usage

```python
import gymnasium as gym
import gym_panda

env = gym.make("gym_panda/PandaPickAndPlace-v3", render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # random action
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

## References

* [panda-gym](https://github.com/qgallouedec/panda-gym)
