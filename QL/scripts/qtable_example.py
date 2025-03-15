import numpy as np
import gym

# Create the environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")  # Small gridworld

# Initialize Q-table (states x actions)
Q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Training loop
for episode in range(1000):  # Train for 1000 episodes
    state, _ = env.reset()
    
    done = False
    while not done:
        # Choose action (ε-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q_table[state, :])  # Exploit
        
        # Take action
        next_state, reward, done, _, _ = env.step(action)
        
        # Update Q-value
        Q_table[state, action] = Q_table[state, action] + alpha * (
            reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action]
        )
        
        # Move to next state
        state = next_state

# Print final Q-table
print("Trained Q-Table:")
print(Q_table)
