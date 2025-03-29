# TRAINING OUR QUADRUPED FOR BASIC WALKING 

Now, after learning some of the basics of RL, let us proceed with how to use RL to train our Quadruped robot. Starting with a very basic example, lets consider to train our robot for satisfying the following objectives:
* Linear velocity only in one direction() = 10 m/s
* No trotting

These are the very two basic cases we will implement initially, and then add more conditions for example maintaining the Torso at some height, some acceleration, gaiting, etc.

For instance, we will be using the A1 Quadruped.

``` python
loadURDF(urdf_path, orientation, quaterion, urdf_flag, fixed_base) : 

# This loads the URDF file in the PyBullet , with the path to the URDF file, the orientation, collision flag and whether the base is fixed(True or False)

getNumJoints(asset) : # Returns the total number of joints in the robot

getJointInfo(asset, joint_id)[2] : # Returns the joint type

getJointInfo(asset, joint_id)[8] : # Returns the lower limit of the joint angle

getJointInfo(asset, joint_id)[9] : # Returns the upper limit of the joint angle

```