import gym
# env to test
# env = gym.make('Ant-v3')
# env = gym.make('HalfCheetah-v3')
# env = gym.make('Hopper-v3')
# env = gym.make('Humanoid-v3')
# env = gym.make('HumanoidStandup-v3')
# env = gym.make('InvertedDoublePendulum-v3')
env = gym.make('InvertedPendulum-v2')
# env = gym.make('Reacher-v3')
# env = gym.make('Swimmer-v3')
# env = gym.make('Walker2d-v3')



# env = gym.make('Walker2d-v3')
# print(env.action_space)
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample() # remplacer par votre algorithme d'apprentissage par renforcement
    observation, reward, done, info = env.step(action)

env.close()
