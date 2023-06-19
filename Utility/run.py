import numpy as np


def multiObjectiveExpriments(alg,environment,noOfTimesteps,random_state=None):
    obs = np.zeros([noOfTimesteps,environment.get_noObjs()])
    pulled_arms = np.zeros(noOfTimesteps)
    alg.reset()
    rng = np.random.RandomState(random_state)
    for t in range(1,noOfTimesteps):
        armToPull = alg.get_arm_index(rng)
        reward    = environment.pull_arm(armToPull,t)
        alg.update_reward(reward)
        obs[t] = reward
        pulled_arms[t] = armToPull
    #if(environment.Nonstationary):
    #    return (obs,pulled_arms,alg.CDsPoints) 
    return (obs,pulled_arms)
