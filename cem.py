import gym
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time

name = 'CartPole-v1'
env = gym.make(name)
observation_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = observation_size * action_size

iterations = 1000 #amount of updates to be performed
population = mp.cpu_count() #amount of parameters to keep track of
keep_count = 2 #the best keep_count thetas are used for the future
max_timesteps = 500 #maximum amount of steps in an environment. In CartPole-v1, this is 500.

maxScore = float('-inf')
maxWeights = None

theta = np.array([hidden_size]) #4 inputs, two outputs (observation and action)
mean = np.random.randn(hidden_size) #the mean for the gaussian distribution
stdev = np.random.randn(hidden_size) #standard deviation for gaussian distribution

def preprocess(state):
    #Preprocesses the state into a horizontal vector.
    return np.reshape(state, [1,observation_size])

def work(e):
    env = gym.make(name)
    s = env.reset()
    d = False
    rewards = 0
    W = np.reshape(e,[observation_size,action_size]) #reshapes the parameter vector to compute actions
    for t in range(max_timesteps):
        #env.render()
        a = np.argmax(np.matmul(s,W))
        ns, r, d, _ = env.step(a)
        s = preprocess(ns)
        rewards += r
        if d:
            return [e,rewards]
    if t == max_timesteps and not d:
        return [e,rewards]

total_rewards = []
now = time.time()
pool = mp.Pool(processes=mp.cpu_count())
for i in range(iterations):
    diag = np.diag(stdev) #diagonal matrix of stdev
    theta = np.random.multivariate_normal(mean,diag,population) #samples population parameter sets given the current mean and standard deviation
    results = [] #results of each rollout
    best = [] #takes the keep_count best thetas
    tasks = []
    for e in theta:
        tasks.append(pool.apply_async(work, (e,)))
    for task in tasks:
        results.append(task.get())
    print 'Episode ' + str(i) +' finished with reward ' + str(results[np.argmax(np.asarray(results)[:,1])][1])
    maximum_result = results[np.argmax(np.asarray(results)[:,1])][1]
    if maximum_result > maxScore:
        maxWeights = results[np.argmax(np.asarray(results)[:,1])][0]
        maxScore = maximum_result
    total_rewards.append(maximum_result) #saves the reward to graph later
    for b in range(keep_count): #takes the best keep_count thetas
        best.append(results[np.argmax(np.asarray(results)[:,1])][0]) #add the best one to a new list
        results.pop(np.argmax(np.asarray(results)[:,1])) #take out the best one from the old list 
    #Creates the new mean and standard deviation based off of our best samples
    mean = np.mean(best,axis=0)
    stdev = np.std(best,axis=0)

np.savez('cartpole.npz', maxWeights)    

print 'Elapsed time ' + str(time.time() - now)
plt.plot(total_rewards)
plt.show()
