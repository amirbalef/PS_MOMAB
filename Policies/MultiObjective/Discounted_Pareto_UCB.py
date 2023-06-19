import numpy as np
class Discounted_Pareto_UCB:
    def __init__(self,noOfObjs ,noOfArms, gamma,  alpha = 2):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.alpha = alpha
        self.noOfArms = noOfArms
        self.noOfObjs = noOfObjs
        self.cummReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.count =  np.zeros(self.noOfArms)
        self.ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.gamma = gamma
        self.ParetoFront = []

    def get_arm_index(self,rng=None):
        if(np.any(self.count== 0)):
            self.lastPlayedArm = np.random.choice(np.argwhere(self.count == 0).reshape(-1))
        else:
            self.ParetoFront = np.ones(self.noOfArms)
            for j in range(self.noOfArms):
                for i in range(self.noOfArms):
                    if(np.all(self.ucb[j] >= self.ucb[i]) and np.any(self.ucb[j] > self.ucb[i])):
                        self.ParetoFront[i] = 0
            if(rng == None):
                self.lastPlayedArm =  np.random.choice(np.where(self.ParetoFront == 1)[0])
            else:
                self.lastPlayedArm =  rng.choice(np.where(self.ParetoFront == 1)[0])
        
        return self.lastPlayedArm

    def update_reward(self,r):
        #self.cummReward[self.lastPlayedArm,:] = (self.cummReward[self.lastPlayedArm,:])*self.gamma +  r
        #self.count[self.lastPlayedArm] = (self.count[self.lastPlayedArm]) *self.gamma +  1
        self.noOfSteps = (self.noOfSteps)*self.gamma  + 1
        
        ParetoFrontSize = self.noOfArms  if np.sum(self.ParetoFront) == 0 else np.sum(self.ParetoFront) 
        
        for i in range(self.noOfArms):
            self.cummReward[i,:] = (self.cummReward[i,:])*self.gamma +  r * (i==self.lastPlayedArm)
            self.count[i] = (self.count[i]) *self.gamma +  1* (i==self.lastPlayedArm)
            if(self.count[i]>0):    
                padding_function = 2 * np.sqrt(self.alpha * np.log(self.noOfSteps *(self.noOfObjs * ParetoFrontSize )**0.25 )) / self.count[i]
                self.ucb[i] = self.cummReward[i] / self.count[i] + padding_function
        
        

    def reset(self):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.cummReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.count =  np.zeros(self.noOfArms)
        self.ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.ParetoFront = []
    
    def info_str(self,latex=0):
        if(latex):
            return "Discounted Pareto UCB \\cite{anquise2021multi} " 
        else:
            return f"Discounted Pareto UCB"