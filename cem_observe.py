import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make('CartPole-v1')
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = observation_size * action_size

iterations = 2 #amount of rollouts to be performed
max_timesteps = 500 #maximum amount of steps in an environment. In CartPole-v1, this is 500.

#env = gym.wrappers.Monitor(env, './recordings', force=True, video_callable=lambda episode_id: True)

theta = np.load('cartpole.npz')['arr_0'] #4 inputs, two outputs (observation and action)
W = np.reshape(theta,[observation_size,action_size]) #reshapes the parameter vector to compute actions

def preprocess(state):
    #Preprocesses the state into a horizontal vector.
    return np.reshape(state, [1,observation_size])

total_rewards = []
for i in range(iterations):
    s = preprocess(env.reset()) #lets our feedforward network manipulate the state
    d = False
    rewards = 0
    for t in range(max_timesteps):
        env.render()
        time.sleep(0.01)
        a = np.argmax(np.matmul(s,W))
        ns, r, d, _ = env.step(a)
        s = preprocess(ns)
        rewards += r
        if d:
            total_rewards.append(rewards)
            break
    print 'Episode ' + str(i) +' finished with reward ' + str(rewards)
    #Creates the new mean and standard deviation based off of our best samples

plt.plot(total_rewards)
plt.show()
