import gym
import numpy as np
import matplotlib.pyplot as plt
## Initialize the environment
# env = gym.make("MountainCar-v0", render_mode="")
env = gym.make("MountainCar-v0")
# Q-learning hyperparameters
LEARNING_RATE = 0.7
DISCOUNT = 0.92
EPISODES = 4000

SHOW_EVERY = 200  # Show every 500 episodes

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
ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

# Function to convert continuous state to discrete state
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

# Training loop
for episode in range(EPISODES):
    episode_reward = 0 
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
        episode_reward += reward 
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
    
    ep_rewards.append(episode_reward)
    ## Taking the verage of the rewards
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")


env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
