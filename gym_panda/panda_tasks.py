from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from gym_panda.pybullet import PyBullet
from gym_panda.robots.panda import Panda
from gym_panda.tasks.pick_and_place import PickAndPlace
from gym_panda.tasks.stack import Stack


def make_env_task(simulator: PyBullet, task_name: str):
    if task_name == "PickAndPlace":
        task = PickAndPlace(simulator)
    elif task_name == "Stack":
        task = Stack(simulator)
    else:
        raise NotImplementedError(task_name)
    return task


class PandaEnv(gym.Env):
    """Robotic task goal env, as the junction of a task and a robot.

    Args:
        task (str): Task name. Can be either "PickAndPlace", "Stack".
        obs_type (str, optional): Observation type. Can be either "state", "pixels" or "pixels_agent_pos".
            Default is "state".
        observation_width (int, optional): Width of the observed image. Defaults to 640.
        observation_height (int, optional): Height of the observed image. Defaults to 640.
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Width of the visualized image. Defaults to 640.
        render_height (int, optional): Height of the visualized image. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targeting this position, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Roll of the camera. Defaults to 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        task: str,
        obs_type: str = "state",
        observation_width: int = 640,
        observation_height: int = 480,
        control_type: str = "ee",
        render_mode: str = "rgb_array",
        renderer: str = "Tiny",
        render_width: int = 640,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        self.sim = PyBullet(render_mode=render_mode, renderer=renderer) # dt = 20/500
        self.task = make_env_task(self.sim, task)
        self.robot = Panda(self.sim, control_type=control_type)         # with gripper
        assert self.robot.sim == self.task.sim, "The robot and the task must belong to the same simulation."

        # Initialize parameters
        self.obs_type = obs_type
        self.observation_width = observation_width
        self.observation_height = observation_height

        self.metadata["render_fps"] = 1 / self.sim.dt
        self.render_width = render_width
        self.render_height = render_height
        self.render_target_position = (
            render_target_position if render_target_position is not None else np.array([0., 0., 0.])
        )
        self.render_distance = render_distance
        self.render_yaw = render_yaw
        self.render_pitch = render_pitch
        self.render_roll = render_roll

        # Initialize spaces
        self.observation_space = self._initialize_observation_space()
        self.action_space = self.robot.action_space

        # Set the view angle
        with self.sim.no_rendering(): 
            self.sim.place_visualizer(
                target_position=self.render_target_position,
                distance=self.render_distance,
                yaw=self.render_yaw,
                pitch=self.render_pitch,
            )

    def _initialize_observation_space(self) -> Dict[str, np.ndarray]:
        image_shape = (self.observation_height, self.observation_width, 3)
        obs = self._get_obs()
        if self.obs_type == "state":
            obs_space = gym.spaces.Dict(
                {
                    "agent_pos": gym.spaces.Box(low=-10., high=10., shape=obs["agent_pos"].shape, dtype=np.float32),
                    "object_pos": gym.spaces.Box(low=-10., high=10., shape=obs["object_pos"].shape, dtype=np.float32),
                }
            )
        elif self.obs_type == "pixels":
            raise NotImplementedError()
        elif self.obs_type == "pixels_agent_pos":
            obs_space = gym.spaces.Dict(
                {
                    "pixels": gym.spaces.Box(low=0, high=255, shape=image_shape, dtype=np.uint8),
                    "agent_pos": gym.spaces.Box(low=-10., high=10., shape=obs["agent_pos"].shape, dtype=np.float32),
                }
            )
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, pixels_agent_pos]"
            )
        return obs_space

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_obs = self.robot.get_obs().astype(np.float32) # joint/ee position, gripper state
        task_obs = self.task.get_obs().astype(np.float32)   # target position, object position
        pixels = self._render()                             # RGB images

        if self.obs_type == "state":
            return {
                "agent_pos": robot_obs,
                "object_pos": task_obs,
            }
        if self.obs_type == "pixels":
            raise NotImplementedError()
        elif self.obs_type == "pixels_agent_pos":
            return {
                "pixels": pixels,
                "agent_pos": robot_obs,
            }
        else:
            raise ValueError(
                f"Unknown obs_type {self.obs_type}. Must be one of [pixels, state, pixels_agent_pos]"
            )
        
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.task.np_random = self.np_random
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        info = {"is_success": bool(self.task.is_success())}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated if the agent has reached the target
        terminated = bool(self.task.is_success())
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward())
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def render(self) -> Optional[np.ndarray]:
        return self._render(visualize=True)
    
    def _render(self, visualize=False) -> Optional[np.ndarray]:
        """Render (RGB images).

        If render mode is "rgb_array", return an RGB array of the scene. Else, do nothing and return None.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        width, height = (
            (self.render_width, self.render_height) 
            if visualize 
            else (self.observation_width, self.observation_height)
        )

        # TODO: render several RGB images
        image = self.sim.render(
            width=width, height=height,
            target_position=self.render_target_position,
            distance=self.render_distance,
            yaw=self.render_yaw,
            pitch=self.render_pitch,
            roll=self.render_roll,
        )
        return image
    
    def save_state(self) -> int:
        """Save the current state of the environment. Restore with `restore_state`.

        Returns:
            int: State unique identifier.
        """
        state_id = self.sim.save_state()
        return state_id

    def restore_state(self, state_id: int) -> None:
        """Restore the state associated with the unique identifier.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.restore_state(state_id)

    def remove_state(self, state_id: int) -> None:
        """Remove a saved state.

        Args:
            state_id (int): State unique identifier.
        """
        self.sim.remove_state(state_id)
