from typing import Optional
from scipy.spatial.transform import Rotation

import gymnasium as gym
import numpy as np

from gym_panda.robots.robot_base import PyBulletRobot
from gym_panda.pybullet import PyBullet
import gym_panda.utils as utils


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optional): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "ee",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.block_gripper = block_gripper
        self.control_type = control_type
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = gym.spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.left_finger = 9
        self.right_finger = 10
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.control_type == "ee":
            ee_displacement = action[:3]
            target_arm_angles = self.ee_to_qdes(ee_displacement)
        else:
            arm_joint_ctrl = action[:7]
            target_arm_angles = self.joint_to_qdes(arm_joint_ctrl)

        if self.block_gripper:
            target_fingers_width = 0
        else:
            fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
            fingers_width = self.get_fingers_width()
            target_fingers_width = fingers_width + fingers_ctrl

        target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
        self.control_joints(target_angles=target_angles)

    def ee_to_qdes(self, ee_displacement: np.ndarray) -> np.ndarray:    # [TODO] consider orientation 
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_pos()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def joint_to_qdes(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles =self.get_joint_pos()
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # joint position / end-effector position
        obs = self.get_ee_pos() if self.control_type == "ee" else self.get_joint_pos()
        observation = obs

        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            observation = np.concatenate((obs, [fingers_width]))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose (9x1)."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_joint_pos(self) -> np.ndarray:
        """Returns the joint position (7x1)"""
        return np.array([self.get_joint_angle(joint=i) for i in range(7)])

    def get_joint_vel(self) -> np.ndarray:
        """Returns the joint velocity (7x1)"""
        return np.array([self.get_joint_velocity(joint=i) for i in range(7)])

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.get_joint_angle(self.fingers_indices[0])
        finger2 = self.get_joint_angle(self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_pos(self) -> np.ndarray:
        """Returns the position of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_vel(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)
    
    def is_grasping(self, object_name: str, min_force=0.5, max_angle=85) -> bool:
        """Check if the robot is grasping an object (ManiSkill/mani_skill/agents/robots/panda/panda.py)

        Args:
            object (string): The object to check if the robot is grasping
            min_force (float, optional): Minimum force before the robot is considered to be grasping the object in Newtons. Defaults to 0.5.
            max_angle (int, optional): Maximum angle of contact to consider grasping. Defaults to 85.
        """
        # Contact force pointing from the object towards the finger
        l_contact_force = self.get_contact_force(self.fingers_indices[0], object_name)
        r_contact_force = self.get_contact_force(self.fingers_indices[1], object_name)
        lforce = np.linalg.norm(l_contact_force)
        rforce = np.linalg.norm(r_contact_force)

        # Direction to open the gripper (y-axis)
        ldirection = Rotation.from_quat(self.get_link_orientation(self.left_finger)).as_matrix()
        rdirection = Rotation.from_quat(self.get_link_orientation(self.right_finger)).as_matrix()
        langle = utils.compute_angle_between(ldirection[:3, 1], l_contact_force)
        rangle = utils.compute_angle_between(- rdirection[:3, 1], r_contact_force)

        # Flag
        lflag = np.logical_and(lforce >= min_force, np.rad2deg(langle) <= max_angle)
        rflag = np.logical_and(rforce >= min_force, np.rad2deg(rangle) <= max_angle)

        return np.logical_and(lflag, rflag)
