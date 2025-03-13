import gym
import numpy as np
## Initializing the Q-Table

env =gym.make("MountainCar-v0", render_mode="human")
env.reset()


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)
## Preparing the Q-table for the table
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)  #REal RL Agent will ot have this hardcoded and it will depend on the env.
## Separation in 20 Buckets
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# Random Q table the values of which will be tweaked by the exploration and exploitation actions of the agent.

print(q_table.shape)
print(q_table)