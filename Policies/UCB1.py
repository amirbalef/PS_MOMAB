"""
    UCB1 Implementation
    Based on: Figure-1, Auer, P., Bianchi, N. C., & Fischer, P. (2002). Finite time analysis of the multiarmed bandit problem. Machine Learning, 47, 235–256.
"""
import numpy as np
class UCB1:
    def __init__(self,noOfArms,  alpha = 2):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.alpha = alpha
        self.noOfArms = noOfArms
        self.cummReward =  np.zeros(self.noOfArms)
        self.count =  np.zeros(self.noOfArms)
        self.ucbIndices =  np.zeros(self.noOfArms)

    def get_arm_index(self):
        if(np.any(self.count== 0)):
            self.lastPlayedArm = np.random.choice(np.argwhere(self.count == 0).reshape(-1))
        else:
            self.lastPlayedArm =  np.argmax(self.ucbIndices)
        return self.lastPlayedArm

    def update_reward(self,r):
        self.cummReward[self.lastPlayedArm] += r
        self.count[self.lastPlayedArm] += 1
        self.noOfSteps += 1
        self.ucbIndices[self.lastPlayedArm] = self.cummReward[self.lastPlayedArm] / self.count[self.lastPlayedArm] + np.sqrt(self.alpha * np.log(self.noOfSteps)) / self.count[self.lastPlayedArm]

    def reset(self):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.cummReward =  np.zeros(self.noOfArms)
        self.count =  np.zeros(self.noOfArms)
        self.ucbIndices =  np.zeros(self.noOfArms)
    
    def info_str(self,latex=0):
            return  f"UCB1"
