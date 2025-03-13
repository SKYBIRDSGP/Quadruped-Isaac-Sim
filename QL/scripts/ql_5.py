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

# Epsilon-greedy parameters
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Initialize Q-table with random values
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Function to convert continuous state to discrete state
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Training loop
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}, Epsilon: {epsilon:.3f}")
        render = True
    else:
        render = False

    # Reset environment and get initial state
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)

    done = False

    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])  # Exploitation
        else:
            action = np.random.randint(0, env.action_space.n)  # Exploration

        # Take action in environment
        new_state, reward, done, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        # Reward shaping for higher speed
        reward += abs(new_state[1]) * 10  # Encourage fast movement

        if render:
            env.render()

        # Q-learning update
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"Reached flag in episode {episode}!")
            q_table[discrete_state + (action,)] = 100  # Higher reward for success

        discrete_state = new_discrete_state

    #  Decay epsilon gradually
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    
    Q_table = q_table

env.close()
