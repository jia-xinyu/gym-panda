import time
import numpy as np
from gym_panda.robots.panda import Panda
from gym_panda.pybullet import PyBullet

def test_kinematics():
    control_type = "ee"
    sim = PyBullet()
    robot = Panda(sim, control_type=control_type)

    # Before
    robot.set_joint_neutral()
    q0 = robot.get_joint_pos()
    print(f"original angles = {q0}")
    pose1 = robot.get_ee_pose()
    print(f"position, orientation = {pose1}")

    # After
    twist = np.array([0., 0.05*20., 0., 0., 0., 0.])  # constraints
    qdes = robot.ee_to_qdes(twist)  # inverse kinematics
    print(f"joint angles = {qdes}")

    robot.control_joints(np.concatenate((qdes, [0.,0.])))
    for _ in range(20):
        robot.sim.step()
    pose2 = robot.get_ee_pose()
    print(f"position, orientation = {pose2}")

    # time.sleep(10)

