import torch
import pybullet as p
import pybullet_data
import numpy as np
import random
import os
import time
import ppo as ppo
from collections import deque

LOAD_MODEL = False
MAX_EPISODES = 1_000_000
MAX_STEPS = 2000
QUAD_MOVE_DEQUE = 10

HM_RANDOM_EPISODES = 10
MAX_TOTAL_REWARD = 200

joint_ids = [ 2,  3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16 ]

###############################################################################################
"""
Given below are the joint IDs for accessing the joints of the four legs of the Quadruped robot

Front_Right : [ 2 , 3 , 4 ]

Front_Left : [ 6 , 7 , 8 ]

Rear_Right : [ 10 , 11 , 12 ]

Rear_Left : [ 14 , 15 , 16 ]"""
###############################################################################################


state_dim = 12
action_dim = 12 
max_action = 0.9

policy = ppo.PPO(state_dim, action_dim, max_action)


# Setting up the Simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf",[0,0,0.0],[0,0,0,1])
p.setGravity(0,0,-9.81)
p.setTimeStep(1./500)

###############################################################################################

for _ep in range(MAX_EPISODES):
    if _ep < HM_RANDOM_EPISODES:
        do_random = True
    else:
        do_random = False
    
    if do_random : print("Random Episode running !")

    # Loading the URDF and setting up the credentials
    urdfFlags = p.URDF_USE_SELF_COLLISION
    urdf_path = os.path.join(os.path.dirname(__file__), "../quad_description/urdf/a1.urdf")

    robot = p.loadURDF(urdf_path, [0,0,0.48],[0,0,0,1], flags = urdfFlags, useFixedBase = False)

    print('EPISODE :- ', _ep)

    quad_starting_pose = []
    quad_prev_rewards = []
    quad_states = []
    quad_actions = []
    quad_action_history = []

    for step in range(MAX_STEPS):

    ## Tracking the Linear and angular velocity of the Quadruped
        linear_vel, anguar_vel = p.getBaseVelocity(robot)
        linear_vel = np.array(linear_vel)
        anguar_vel = np.array(anguar_vel)

        if step == 0 :
            quad_prev_rewards = 0
            quad_action_history = deque(maxlen=QUAD_MOVE_DEQUE)
        
        observation = np.float32(p.getJointState(robot, joint) for joint in joint_ids)
        quad_states = np.float32(observation)

        if not do_random and _ep % 10 ==0 :
            # Exploitation in the PPO
            action = policy.select_action(observation)
        else:
            if do_random:
                action = np.random.randn(action_dim).clip(-max_action, max_action)
            else:
                action = policy.select_action(observation)
            
        quad_actions = action
        quad_action_history.append(action)

        for i in range(joint_ids):
            p.setJointMotorControlArray(robot, [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16], 
                                p.POSITION_CONTROL, 
                                targetPositions = action[i],
                                forces = [50] * len(joint_ids))

        new_lin_vel, new_ang_vel = p.getBaseVelocity(robot) 
        new_lin_vel = np.float32(new_lin_vel)

        new_observation = np.float32(p.getJointState(robot, joint) for joint in joint_ids)

        start_x_vel = linear_vel[0]
        start_y_vel = linear_vel[1]
        start_z_vel = linear_vel[2]

        current_x_vel = new_lin_vel[0]
        current_y_vel = new_lin_vel[1]
        current_z_vel = new_lin_vel[2]

        

    


