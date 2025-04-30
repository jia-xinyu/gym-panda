from abc import ABC, abstractmethod
from typing import Optional

import pinocchio as pin
import gymnasium as gym
import numpy as np

from example_robot_data import load
from gym_panda.pybullet import PyBullet
from gym_panda.robots.kinematics import pinKinematics


class PyBulletRobot(ABC):
    """Base class for robot env.

    Args:
        sim (PyBullet): Simulation instance.
        urdf_name (str): Robot name in example-robot-data.
        tcp_name (str): Name of the tool center point.
        base_position (np.ndarray): Position of the base of the robot as (x, y, z).
        base_orientation (np.ndarray): Euler orientation of the robot, as (rx, ry, rz).
        joint_indices (np.ndarray): Joint indices including fingers.
        joint_forces (np.ndarray): Maximum motor force used to reach the target value.
        lock_pinocchio_fingers (list, optional): locked fingers in Pinocchio.
    """

    def __init__(
        self,
        sim: PyBullet,
        urdf_name: str,
        tcp_name: str,
        base_position: np.ndarray,
        base_orientation: np.ndarray,
        action_space: gym.spaces.Space,
        joint_indices: np.ndarray,
        joint_forces: np.ndarray,
        lock_pinocchio_joints: Optional[list] = None,
    ) -> None:
        self.sim = sim
        self.body_name = urdf_name
        robot = load(urdf_name)
        model = robot.model
        if lock_pinocchio_joints:
            model = pin.buildReducedModel(model, lock_pinocchio_joints, np.zeros(model.nq))
        data = model.createData()
        with self.sim.no_rendering():
            self._load_robot(robot.urdf, base_position, base_orientation)
            self.setup()
        self.action_space = action_space
        self.joint_indices = joint_indices
        self.joint_forces = joint_forces
        self.kin_solver = pinKinematics(model, data, model.getFrameId(tcp_name))

    def _load_robot(self, file_name: str, base_position: np.ndarray, base_orientation: np.ndarray) -> None:
        """Load the robot.

        Args:
            file_name (str): URDF file name of the robot.
            base_position (np.ndarray): Position of the robot, as (x, y, z).
            base_orientation (np.ndarray): Euler orientation of the robot, as (rx, ry, rz).
        """
        self.sim.loadURDF(
            body_name=self.body_name,
            fileName=file_name,
            basePosition=base_position,
            baseOrientation=self.sim.physics_client.getQuaternionFromEuler(base_orientation),
            useFixedBase=True,
        )

    def setup(self) -> None:
        """Called after robot loading."""
        pass

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().

        Args:
            action (np.ndarray): Action.
        """

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: Observation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""

    def get_link_position(self, link: int) -> np.ndarray:
        """Returns the position of a link as (x, y, z)

        Args:
            link (int): Link index.

        Returns:
            np.ndarray: Position as (x, y, z).
        """
        return self.sim.get_link_position(self.body_name, link)

    def get_link_orientation(self, link: int) -> np.ndarray:
        """Returns the orientation of a link as (qx, qy, qz, qw)

        Args:
            link (int): Link index.

        Returns:
            np.ndarray: Orientation as (qx, qy, qz, qw).
        """
        return self.sim.get_link_orientation(self.body_name, link)
    
    def get_link_linear_velocity(self, link: int) -> np.ndarray:
        """Returns the linear velocity of a link as (vx, vy, vz)

        Args:
            link (int): Link index.

        Returns:
            np.ndarray: Linear velocity as (vx, vy, vz).
        """
        return self.sim.get_link_linear_velocity(self.body_name, link)

    def get_link_angular_velocity(self, link: int) -> np.ndarray:
        """Returns the angular velocity of a link as (wx, wy, wz)

        Args:
            link (int): Link index.

        Returns:
            np.ndarray: Angular velocity as (wx, wy, wz).
        """
        return self.sim.get_link_angular_velocity(self.body_name, link)

    def get_joint_angle(self, joint: int) -> float:
        """Returns the angle of a joint

        Args:
            joint (int): Joint index.

        Returns:
            float: Joint angle
        """
        return self.sim.get_joint_angle(self.body_name, joint)

    def get_joint_velocity(self, joint: int) -> float:
        """Returns the velocity of a joint as (wx, wy, wz)

        Args:
            joint (int): Joint index.

        Returns:
            float: Joint velocity.
        """
        return self.sim.get_joint_velocity(self.body_name, joint)

    def control_joints(self, target_angles: np.ndarray) -> None:
        """Control the joints of the robot.

        Args:
            target_angles (np.ndarray): Target angles. The length of the array must equal to the number of joints.
        """
        self.sim.control_joints(
            body=self.body_name,
            joints=self.joint_indices,
            target_angles=target_angles,
            forces=self.joint_forces,
        )

    def set_joint_angles(self, angles: np.ndarray) -> None:
        """Set the joint position of a body. Can induce collisions.

        Args:
            angles (np.ndarray): Joint angles.
        """
        self.sim.set_joint_angles(self.body_name, joints=self.joint_indices, angles=angles)
        
    def forward_kinematics(self, angles: np.ndarray) -> pin.SE3:
        """Compute forward kinematics. 

        Args:
            angles (np.ndarray): Joint angles.

        Returns:
            SE3: Tip frame placement.
        """
        return self.kin_solver.compute_fk(q=angles)

    def inverse_kinematics(self, target_pose: pin.SE3, angles: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            target_pose (pin.SE3): Target pose.
            angles (np.ndarray): Joint angles.

        Returns:
            np.ndarray: List of joint angles.
        """
        target_velocity = self.kin_solver.compute_ik(Tdes=target_pose, q=angles, dt=self.sim.dt)
        target_angles = angles + target_velocity * self.sim.dt

        return target_angles

    '''
    def inverse_kinematics(self, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint values.

        Args:
            link (int): The link.
            position (x, y, z): Desired position of the link.
            orientation (x, y, z, w): Desired orientation of the link.

        Returns:
            List of joint values.
        """
        inverse_kinematics = self.sim.inverse_kinematics(self.body_name, link=link, position=position, orientation=orientation)
        return inverse_kinematics
    '''

    def get_contact_force(self, link: int, object_name: str) -> np.ndarray:
        """Get the contact force between a link and an object. 

        Args:
            link (int): Link index in the body.
            object_name (str): Object unique name.
        
        Returns:
            np.ndarray: Force as (x, y, z). Force direction pointing from the object towards the body.
        """
        contact_force = self.sim.get_contact_force(bodyA=self.body_name, bodyB=object_name, linkA=link, linkB=-1)

        return contact_force
        
