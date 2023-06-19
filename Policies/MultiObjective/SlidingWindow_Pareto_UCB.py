import numpy as np
class SlidingWindow_Pareto_UCB:
    def __init__(self,noOfObjs ,noOfArms, omega, gamma = 1.0,  alpha = 2):
        self.t = 0
        self.lastPlayedArm = 0
        self.PlayedArms = []
        self.Rewards = []
        self.alpha = alpha
        self.noOfArms = noOfArms
        self.noOfObjs = noOfObjs
        self.meanReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.sw_count =  np.zeros(self.noOfArms)
        self.count =  np.zeros(self.noOfArms)
        self.ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.gamma = gamma
        self.omega = omega
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
        
        self.PlayedArms.append(self.lastPlayedArm)
        return self.lastPlayedArm

    def update_reward(self,r):
        self.t += 1
        self.Rewards.append(r)
        ParetoFrontSize = self.noOfArms  if np.sum(self.ParetoFront) == 0 else np.sum(self.ParetoFront) 
        
        
        self.count[self.lastPlayedArm] = np.sum(self.lastPlayedArm==np.array(self.PlayedArms[-self.omega:]))
        if(self.count[self.lastPlayedArm]>0):
            if(self.t -self.omega >= 0 ):
                discount = np.ones(self.omega) * self.gamma**(np.arange(self.omega)[::-1])
                multi_objective_discount = np.tile(discount.reshape(-1,1),(1,self.noOfObjs))
                n_t = np.sum([ np.sum(discount[self.lastPlayedArm==np.array(self.PlayedArms[-self.omega:])])  for i in range(self.noOfArms)])
                N = np.sum(discount[self.lastPlayedArm==np.array(self.PlayedArms[-self.omega:])] )
                R = (1/N) * np.sum( (multi_objective_discount * np.array(self.Rewards[-self.omega:]))[self.lastPlayedArm==np.array(self.PlayedArms[-self.omega:])], axis = 0 )
            else:
                discount = np.ones(self.t) * self.gamma**(np.arange(self.t)[::-1])
                multi_objective_discount = np.tile(discount.reshape(-1,1),(1,self.noOfObjs))
                n_t = np.sum(discount)
                N = np.sum(discount[self.lastPlayedArm==np.array(self.PlayedArms)] )
                R = (1/N) * np.sum( (multi_objective_discount * np.array(self.Rewards))[self.lastPlayedArm==np.array(self.PlayedArms)], axis = 0 )
            
            self.sw_count[self.lastPlayedArm] = N
            self.meanReward[self.lastPlayedArm,:] = R
            for i in range(self.noOfArms):
                if(self.sw_count[i]> 0):
                    padding_function = np.sqrt(self.alpha * np.log(n_t *(self.noOfObjs * ParetoFrontSize )**0.25 )) /  self.sw_count[i]
                    self.ucb[i] = self.meanReward[i] + padding_function
                
        

    def reset(self):
        self.t = 0
        self.lastPlayedArm = 0
        self.PlayedArms = []
        self.Rewards = []
        self.meanReward =  np.zeros([self.noOfArms,self.noOfObjs])
        self.sw_count =  np.zeros(self.noOfArms)
        self.count =  np.zeros(self.noOfArms)
        self.ucb =  np.zeros([self.noOfArms, self.noOfObjs])
        self.ParetoFront = []
    
    def info_str(self,latex=0):
        if(latex):
            return "SW Pareto UCB \\cite{anquise2021multi}" 
        else:
            return f"SW Pareto UCB"