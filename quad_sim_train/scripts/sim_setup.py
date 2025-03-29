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
###############################################################################################
"""
Given below are the joint IDs for accessing the joints of the four legs of the Quadruped robot

Front_Right : [ 2 , 3 , 4 ]

Front_Left : [ 6 , 7 , 8 ]

Rear_Right : [ 10 , 11 , 12 ]

Rear_Left : [ 14 , 15 , 16 ]"""
###############################################################################################
joint_ids = [ 2,  3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16 ]

joint_limits = []

for i in joint_ids:
    info = p.getJointInfo(robot, i)
    lower_limit = info[8]
    upper_limit = info[9]

    joint_limits.append([i,lower_limit, upper_limit])
###############################################################################################
"""The array stores the joint Id at the 0th index, lower joint limit at the 1st and the 
   upper joint limit in the 2nd index of the joint_limits[]

   Just uncomment this part for any confirmation:

   for i in range(len(joint_ids)):

    print(joint_limits[i][0])
    print(joint_limits[i][1])
    print(joint_limits[i][2])

    print("----------------------")

""" 
###############################################################################################
while True: 

    p.stepSimulation()

    time.sleep(1./500)