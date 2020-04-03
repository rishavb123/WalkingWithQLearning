import gym

# env = gym.make('BipedalWalker-v3')
# env = gym.make('LunarLander-v2')
# env = gym.make('CarRacing-v0')
env = gym.make('CartPole-v1')

env.reset()
print(env.action_space.sample(), len(env.get_state()))
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()