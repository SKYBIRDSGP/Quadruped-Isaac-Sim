import gym
import numpy as np

## Initialize the environment
env = gym.make("MountainCar-v0", render_mode="human")

# Q-learning hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.99
EPISODES = 25000

SHOW_EVERY = 2000  # Show every 2000 episodes

# Discretization parameters
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)  
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


# Initialize Q-table with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Function to convert continuous state to discrete state
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Training loop
for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}")
        render = True
    else:
        render = False

    # Reset environment and get initial state
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)

    done = False

    while not done:
        # Always select the best action (no exploration)
        action = np.argmax(q_table[discrete_state])

        # Take action in environment
        new_state, reward, done, truncated, _ = env.step(action)

        # Convert new state to discrete
        # new_discrete_state = get_discrete_state(new_state)
        new_discrete_state = get_discrete_state(new_state)

        env.render()
        #new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    
        # If simulation did not end yet after last step - update Q table
        if not done:
        
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])
    
            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]
    
            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
    
            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q
    
    
        # Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            #q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0
    
        discrete_state = new_discrete_state


env.close()

        # if render:
        #     env.render()

        # if not done:
        #     # Q-learning update equation
        #     max_future_q = np.max(q_table[new_discrete_state])
        #     current_q = q_table[discrete_state + (action, )]
        #     new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        #     q_table[discrete_state + (action, )] = new_q

        # elif new_state[0] >= env.goal_position:
        #     print(f"We made it in episode {episode}!")
        #     q_table[discrete_state + (action, )] = 0.2  # Directly assign reward for reaching the goal

        # # Update state
        # discrete_state = new_discrete_state

env.close()
