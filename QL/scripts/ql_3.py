import gym
import numpy as np
## Initializing the Q-Table

env =gym.make("MountainCar-v0", render_mode="human")

LEARNING_RATE = 0.05
DISCOUNT = 0.98
# Discount is for how much do you wanna priority the future reward over current reward
EPISODES = 25000

SHOW_EVERY = 2000

## Preparing the Q-table for the table
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)  #REal RL Agent will ot have this hardcoded and it will depend on the env.
## Separation in 20 Buckets
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# Random Q table the values of which will be tweaked by the exploration and exploitation actions of the agent.

# We have to convert the continuous states into discrete states

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    
    if(episode%SHOW_EVERY == 0):
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset()[0])

    # print(discrete_state)

    # print(np.argmax(q_table[discrete_state])) ## getting the max one

    ## Here we go !!!
    done = False

    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, truncated, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()

        if not done:
            max_fucture_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
    ## Calculating all the Q-Values
            new_q = (1 - LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT*max_fucture_q)
            q_table[discrete_state + (action, )] = new_q

        elif new_state[0]>= env.goal_position:
            q_table[discrete_state + (action, )] = 0    ## Reward for completing the thing


        discrete_state = new_discrete_state

env.close()