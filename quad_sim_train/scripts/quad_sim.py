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

robot = p.loadURDF(urdf_path, [0,0,0.48],[0,0,0,1], flags = urdfFlags, useFixedBase = False)
print(urdf_path)

while True: 
    p.stepSimulation()
    time.sleep(1./500)