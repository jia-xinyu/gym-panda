# gym-panda

## Introduction
Based on [panda-gym](https://github.com/qgallouedec/panda-gym), the following features are added:

* reward shaping for manipulation grasping,
* inverse kinematics solver via [Quadprog](https://github.com/quadprog/quadprog),
* support of rigid body algorithms via [Pinocchio](https://github.com/stack-of-tasks/pinocchio), 
* environments of imitation learning algorithms in [LeRobot](https://github.com/huggingface/lerobot),
* environments of reinforcement learning like [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).


## Installation
Create a virtual environment with Python 3.10 and activate it, e.g. with miniconda:

```bash
conda create -y -n gym-panda python=3.10
conda activate gym-panda
```

Download our source code and install:
```bash
git clone https://github.com/jia-xinyu/gym-panda.git && cd gym-panda
pip install -e .
```

[Pinocchio](https://github.com/stack-of-tasks/pinocchio) and [example-robot-data](https://github.com/Gepetto/example-robot-data) are also needed:
```bash
conda install pinocchio example-robot-data -c conda-forge
```

## Usage

* Run the demo file
```bash
python demo.py
```

<div align="center">
<img width="800" src="docs/PandaPickAndPlace.png">
</div>

* Run unit tests
```bash
pytest test/envs_test.py  -rA
```
