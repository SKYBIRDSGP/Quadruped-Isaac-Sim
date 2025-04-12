import pybullet as p
import pybullet_data
import numpy as np
import random
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

    angle_joint_2 = np.random.uniform(joint_limits[0][1] , joint_limits[0][2])
    angle_joint_3 = np.random.uniform(joint_limits[1][1] , joint_limits[1][2])
    angle_joint_4 = np.random.uniform(joint_limits[2][1] , joint_limits[2][2])

    angle_joint_6 = np.random.uniform(joint_limits[3][1] , joint_limits[3][2])
    angle_joint_7 = np.random.uniform(joint_limits[4][1] , joint_limits[4][2])
    angle_joint_8 = np.random.uniform(joint_limits[5][1] , joint_limits[5][2])

    angle_joint_10 = np.random.uniform(joint_limits[6][1] , joint_limits[6][2])
    angle_joint_11 = np.random.uniform(joint_limits[7][1] , joint_limits[7][2])
    angle_joint_12 = np.random.uniform(joint_limits[8][1] , joint_limits[8][2])

    angle_joint_14 = np.random.uniform(joint_limits[9][1] , joint_limits[9][2])
    angle_joint_15 = np.random.uniform(joint_limits[10][1] , joint_limits[10][2])
    angle_joint_16 = np.random.uniform(joint_limits[11][1] , joint_limits[1][2])

    p.setJointMotorControlArray(robot, [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16], 
                                p.POSITION_CONTROL, 
                                targetPositions = [angle_joint_2, angle_joint_3, angle_joint_4, angle_joint_6, angle_joint_7, angle_joint_8, angle_joint_10, angle_joint_11, angle_joint_12, angle_joint_14, angle_joint_15, angle_joint_16],
                                forces = [50] * len(joint_ids))
    
    ## Tracking the Linear and angular velocity of the Quadruped
    linear_vel, anguar_vel = p.getBaseVelocity(robot)
    linear_vel = np.array(linear_vel)
    anguar_vel = np.array(anguar_vel)

    print(linear_vel)
    print(anguar_vel)

    p.stepSimulation()

    time.sleep(1./200)