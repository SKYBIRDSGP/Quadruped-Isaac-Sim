import gym

env =gym.make("MountainCar-v0", render_mode="human")
env.reset()


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)


done = False

while not done:
    action = 2
    new_state, reward, done, truncated, _ = env.step(action)
    print(new_state)
    env.render()

env.close()