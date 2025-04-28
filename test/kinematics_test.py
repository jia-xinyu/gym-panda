import time
import numpy as np
from gym_panda.robots.panda import Panda
from gym_panda.pybullet import PyBullet

def test_kinematics():
    control_type = "ee"
    sim = PyBullet(render_mode="human")
    robot = Panda(sim, control_type=control_type)

    # Before
    robot.set_joint_neutral()
    q0 = robot.get_joint_pos()
    print(f"original angles = {q0}")

    pos = robot.get_ee_pos()
    ori = sim.physics_client.getEulerFromQuaternion(robot.get_ee_ori())
    print(f"position = {pos}, orientation = {ori}")

    # After
    twist = np.array([0., 0., 0., 0., 0., -1.57*20]) # rotate world-z around 90
    qdes = robot.ee_to_qdes(twist)  # random action
    print(f"joint angles = {qdes}")

    robot.control_joints(np.concatenate((qdes, [0.,0.])))
    robot.sim.step()
    pos = robot.get_ee_pos()
    ori = sim.physics_client.getEulerFromQuaternion(robot.get_ee_ori())
    print(f"position = {pos}, orientation = {ori}")

    # q_test = np.array([0., 0., 0., 0., 0., 0., 0])  # from pinochcio
    # assert np.allclose(qdes, q_test)

    time.sleep(10)

