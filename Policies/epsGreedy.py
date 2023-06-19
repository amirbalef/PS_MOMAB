import numpy as np
"""
    ϵ-Greedy Implementation
"""
class epsGreedy:
    def __init__(self,noOfArms,  epsilon = 0.1):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.epsilon = epsilon
        self.noOfArms = noOfArms
        self.cummReward =  np.zeros(self.noOfArms)
        self.count =  np.zeros(self.noOfArms)
        self.avgValue =  np.zeros(self.noOfArms)

    def get_arm_index(self):
        if(np.random.uniform() > self.epsilon) :
            self.lastPlayedArm = np.argmax(self.avgValue)
        else:
            self.lastPlayedArm =  np.random.randint(0,self.noOfArms)
        return self.lastPlayedArm

    def update_reward(self,r):
        self.cummReward[self.lastPlayedArm] += r
        self.count[self.lastPlayedArm] += 1
        self.noOfSteps += 1
        self.avgValue[self.lastPlayedArm] = self.cummReward[self.lastPlayedArm] / self.count[self.lastPlayedArm]

    def reset(self):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.cummReward =  np.zeros(self.noOfArms)
        self.count =  np.zeros(self.noOfArms)
        self.avgValue =  np.zeros(self.noOfArms)
    
    def info_str(self,latex=0):
        if latex:
            return f"\$\\epsilon\$-Greedy (\$\\epsilon = {self.epsilon})"
        else:
            return  f"ϵ-Greedy (ϵ = {self.epsilon})"
