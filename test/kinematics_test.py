import time
import numpy as np
from gym_panda.robots.panda import Panda
from gym_panda.pybullet import PyBullet

def test_kinematics():
    control_type = "ee"
    # sim = PyBullet(render_mode="human")
    sim = PyBullet(render_mode="rgb_array")
    robot = Panda(sim, control_type=control_type)

    # Before
    robot.set_joint_neutral()
    q0 = robot.get_joint_pos()
    print(f"original angles = {q0}")

    pose1 = robot.get_ee_pose()
    pos1 = pose1[:3]
    ori1 = sim.physics_client.getEulerFromQuaternion(pose1[3:])
    print(f"position = {pos1}, orientation = {ori1}")

    # After
    twist = np.array([0., 0., 0., 0., -10*10, 0.]) # rotate z around 90
    qdes = robot.ee_to_qdes(twist)  # random action
    print(f"joint angles = {qdes}")

    robot.control_joints(np.concatenate((qdes, [0.,0.])))
    robot.sim.step()
    pose2 = robot.get_ee_pose()
    pos2 = pose2[:3]
    ori2 = sim.physics_client.getEulerFromQuaternion(pose2[3:])
    print(f"position = {pos2}, orientation = {ori2}")

    # q_test = np.array([0., 0., 0., 0., 0., 0., 0])  # from pinochcio
    # assert np.allclose(qdes, q_test)

    # time.sleep(10)

