import pybullet as p
import pybullet_data
import os
import time

# Setting up the Simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf",[0,0,0.0],[0,0,0,1])
p.setGravity(0,0,-9.81)
p.setTimeStep(1./500)

# Loading the URDF and setting up the credentials
urdfFlags = p.URDF_USE_SELF_COLLISION
urdf_path = os.path.join(os.path.dirname(__file__), "../quad_description/urdf/a1.urdf")

robot = p.loadURDF(urdf_path, [0,0,0.48],[0,0,0,1], flags = urdfFlags, useFixedBase = True)
print(urdf_path)
print("__________________")
num_joints = p.getNumJoints(robot)
print(f"Number of joints: {num_joints}")

for j in range(num_joints):
    joint_info = p.getJointInfo(robot, j)  # Get joint details
    joint_id = joint_info[0]  # Joint index
    joint_name = joint_info[1].decode("utf-8")  # Joint name (needs decoding)
    link_name = joint_info[12].decode("utf-8")  # Link name (needs decoding)
    link = p.getLinkState(robot, j)
    print(f"Joint ID: {joint_id}, Joint Name: {joint_name}, Link Name: {link_name}\n")
    print(link)
while True: 
    p.stepSimulation()
    time.sleep(1./500)