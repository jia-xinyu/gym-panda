import pinocchio as pin
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

# [Pybullet]
def check_pybullet(file_name, joints_idx=None, q=None, gui=False):
    if not gui:
        connection_mode = p.DIRECT  # For RL
    else:
        connection_mode = p.GUI
    sim = bc.BulletClient(connection_mode=connection_mode)
    sim.setGravity(0, 0, -9.81)
    sim.setAdditionalSearchPath(pybullet_data.getDataPath()) # for loading URDF files

    robot = sim.loadURDF(
        fileName=file_name,
        useFixedBase=True,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        flags=p.URDF_USE_SELF_COLLISION)
    
    # Step simulation
    if joints_idx is not None:
        p.setJointMotorControlArray(robot, joints_idx, p.POSITION_CONTROL, targetPositions=q)
        for i in range(20):
            sim.stepSimulation()

    num_joints = p.getNumJoints(robot)
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot, i)
        print(f"Joint Index: {joint_info[0]}")
        print(f"Joint Name: {joint_info[1]}")
        print(f"Joint Type: {joint_info[2]}")
        print(f"Joint Lower Limit: {joint_info[8]}")
        print(f"Joint Upper Limit: {joint_info[9]}")
        link_info = p.getLinkState(robot, i)
        print(f"Link Name: {joint_info[12]}, Link Index: {i}")
        print("Link World Position: ({: .4f}, {: .4f}, {: .4f})".format(*link_info[0]))
        print("Link World Orientation: ({: .4f}, {: .4f}, {: .4f})".format(*link_info[1]))
        print("-----------------------------")


# [Pinocchio]
def check_pinocchio(model, data, collision_model, collision_data, q):
    print("\nJoint configuration:", q)

    # Compute FK
    pin.forwardKinematics(model, data, q)

    # Check collision geometries & frames
    pin.updateGeometryPlacements(model, data,
                                collision_model, collision_data, q)
    pin.updateFramePlacements(model, data)

    print("\nJoint placements:")
    for i in range(len(data.oMi)):
        pos = list(data.oMi[i].translation.T.flat)
        print(("{:<2} {:<24} : {: .4f} {: .4f} {: .4f}"
                .format(i, model.names[i], *pos)))

    print("\nGeometry placements:")
    for i in range(len(collision_data.oMg)):
        pos = list(collision_data.oMg[i].translation.T.flat)
        print(("{:<2} {:<24} : {: .4f} {: .4f} {: .4f}"
                .format(i, collision_model.geometryObjects[i].name, *pos)))

    print("\nFrame placements:")
    for i in range(len(data.oMf)):
        pos = list(data.oMf[i].translation.T.flat)
        print(("{:<2} {:<24} : {: .4f} {: .4f} {: .4f}"
                .format(i, model.frames[i].name, *pos)))

    print("\n------------------------")


if __name__ == '__main__':
    import numpy as np

    # [Pybullet] Load from pybullet_data
    joints_idx = [0, 1, 2, 3, 4, 5, 6, 9, 10]
    # check_pybullet("franka_panda/panda.urdf", joints_idx, np.zeros(len(joints_idx)), gui=False)

    # [Pybullet] Load from example-robot-data
    from example_robot_data import load
    joints_idx = [0, 1, 2, 3, 4, 5, 6, 10, 11]
    robot = load("panda")
    # check_pybullet(robot.urdf, joints_idx, np.zeros(len(joints_idx)), gui=False)


    # [Pinocchio] Load from example-robot-data
    from example_robot_data import load
    robot = load("panda")
    # robot = load("ur5_gripper")
    dof = robot.nq
    model = robot.model
    data = robot.data
    collision_model = robot.collision_model
    collision_data = robot.collision_data
    # check_pinocchio(model, data, collision_model, collision_data, np.zeros(model.nq))

    # [Pinocchio] Load from pybullet_data
    import pybullet_data
    dir = pybullet_data.getDataPath() + "/franka_panda"
    urdf_filename = dir + "/panda.urdf"
    model, collision_model, _ = pin.buildModelsFromUrdf(urdf_filename, dir)
    data = model.createData()
    collision_data = collision_model.createData()
    check_pinocchio(model, data, collision_model, collision_data, np.zeros(model.nq))
