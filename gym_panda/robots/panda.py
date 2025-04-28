from typing import Optional
from scipy.spatial.transform import Rotation as R

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
        control_type (str, optional): "ee" to control end-effector 6D pose or "joints" to control joint angles.
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
        n_action = 6 if self.control_type == "ee" else 7  # control 6D pose if "ee", else, control 7 joints
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
            ee_twist = action[:6]
            target_arm_angles = self.ee_to_qdes(ee_twist)
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

    def ee_to_qdes(self, ee_twist: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector twist in Catersian space.

        Args:
            ee_twist (np.ndarray): Twist, as (vx, vy, vz, wx, wy, wz).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        # Get the target pos
        ee_translation = ee_twist[:3] * 0.05    # limit maximum change in position
        target_ee_pos = self.get_ee_pos() + ee_translation
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_pos[2] = np.max((0, target_ee_pos[2]))

        # Get the target quat
        ee_rotation = ee_twist[3:] * 0.05       # limit maximum change in orientation
        target_ee_ori = R.from_quat(self.get_ee_ori()) * R.from_rotvec(ee_rotation)
        target_ee_ori = target_ee_ori.as_quat()
        # target_ee_ori = np.array([1.0, 0.0, 0.0, 0.0])  # rotate 180 around x-axis

        # Compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_pos, orientation=target_ee_ori
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
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in joint position
        # Get the current position and the target position
        current_arm_joint_angles =self.get_joint_pos()
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector 6D pose / joint angle
        obs = self.get_ee_pose() if self.control_type == "ee" else self.get_joint_pos()
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
        """Returns the position (3x1) of the end-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)
    
    def get_ee_ori(self) -> np.ndarray:
        """Returns the orientation (4x1) of the end-effector as (rx, ry, rz, w)"""
        return self.get_link_orientation(self.ee_link)
    
    def get_ee_pose(self) -> np.ndarray:
        """Returns the pose (7x1) of the end-effector as (x, y, z, rx, ry, rz, w)"""
        pos = self.get_ee_pos()
        ori = self.get_ee_ori()
        # ori = R.from_quat(self.get_ee_ori()).as_euler('xyz', degrees=False)
        return np.hstack((pos, ori))
    
    def get_ee_vel(self) -> np.ndarray:
        """Returns the velocity (6x1) of the end-effector as (vx, vy, vz, wx, wy, wz)"""
        v = self.get_link_linear_velocity(self.ee_link)
        w = self.get_link_angular_velocity(self.ee_link)
        return np.hstack((v, w))
        
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
        ldirection = R.from_quat(self.get_link_orientation(self.left_finger)).as_matrix()
        rdirection = R.from_quat(self.get_link_orientation(self.right_finger)).as_matrix()
        langle = utils.compute_angle_between(ldirection[:3, 1], l_contact_force)
        rangle = utils.compute_angle_between(- rdirection[:3, 1], r_contact_force)

        # Flag
        lflag = np.logical_and(lforce >= min_force, np.rad2deg(langle) <= max_angle)
        rflag = np.logical_and(rforce >= min_force, np.rad2deg(rangle) <= max_angle)

        return np.logical_and(lflag, rflag)
