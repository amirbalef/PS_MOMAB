import numpy as np
from Policies.MultiObjective.utils.kl_distance_function import kl_distance_function


class Pareto_KL_UCB:
    def __init__(self,noOfObjs ,noOfArms):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.noOfArms = noOfArms
        self.noOfObjs = noOfObjs
        self.cummReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.count =  np.zeros(self.noOfArms)
        self.kl_ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.ParetoFront = []
        self.kl_distance = kl_distance_function()

    def get_arm_index(self,rng=None):
        if(np.any(self.count== 0)):
            self.lastPlayedArm = np.random.choice(np.argwhere(self.count == 0).reshape(-1))
        else:
            self.ParetoFront = np.ones(self.noOfArms)
            for j in range(self.noOfArms):
                for i in range(self.noOfArms):
                    if(np.all(self.kl_ucb[j] >= self.kl_ucb[i]) and np.any(self.kl_ucb[j] > self.kl_ucb[i])):
                        self.ParetoFront[i] = 0
            if(rng == None):
                self.lastPlayedArm =  np.random.choice(np.where(self.ParetoFront == 1)[0])
            else:
                self.lastPlayedArm =  rng.choice(np.where(self.ParetoFront == 1)[0])
        return self.lastPlayedArm
        
    def update_reward(self,r):
        self.cummReward[self.lastPlayedArm,:] += r
        self.count[self.lastPlayedArm] += 1
        self.noOfSteps += 1

        for i in range(self.noOfArms):
            if(self.count[i]>0):
                d =   np.log( 1 + self.noOfSteps )/ self.count[i]
                p = self.cummReward[i]/ self.count[i]
                q = self.kl_distance.compute_max_value(p,d)
                self.kl_ucb[i] = q 

    def reset(self):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.cummReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.count =  np.zeros(self.noOfArms)
        self.kl_ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.ParetoFront = []
    
    def info_str(self,latex=0):
        if(latex):
            return "\\textbf{{Pareto klUCB}}" 
        else:
            return "Pareto klUCB" 