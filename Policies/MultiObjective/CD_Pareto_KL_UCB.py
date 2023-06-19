import numpy as np
from Policies.MultiObjective.utils.kl_distance_function import kl_distance_function

class CD_Pareto_KL_UCB:
    def __init__(self,CDs ,noOfObjs ,noOfArms, steps_to_explore):
        self.noOfSteps     = 0
        self.lastPlayedArm = 0
        self.noOfArms = noOfArms
        self.noOfObjs = noOfObjs
        self.cummReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.count =  np.zeros(self.noOfArms)
        self.counter =  np.zeros(self.noOfArms)
        self.kl_ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.CDs = CDs
        self.lamda     = steps_to_explore
        self.kl_distance = kl_distance_function()
        
    def get_arm_index(self,rng=None):    #Choose the best arm for the policy
        if(np.any(self.counter== 0)):
            self.lastPlayedArm = np.random.choice(np.argwhere(self.counter == 0).reshape(-1))
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
        self.CDs.t += 1
        if (self.CDs.update(self.lastPlayedArm,r) != 0):
            self.reset()
            
        if(self.noOfSteps % self.lamda == 0 ):
            self.counter =  np.zeros(self.noOfArms)
            
        self.cummReward[self.lastPlayedArm,:] += r
        self.count[self.lastPlayedArm] += 1
        self.counter[self.lastPlayedArm] += 1
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
        self.counter =  np.zeros(self.noOfArms)
        self.kl_ucb =  np.zeros([self.noOfArms, self.noOfObjs])
    
    def info_str(self,latex=0):
        if(latex):
            return "\\textbf{{"+ self.CDs.info_str(latex) + " Pareto klUCB}}" 
        else:
            return self.CDs.info_str(latex) + f" Pareto klUCB" 
