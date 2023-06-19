import numpy as np

class MultiObjectiveEnvironment:
    def __init__(self, arms, time_horizon, Nonstationary):
        self.time_horizon = time_horizon
        self.arms = arms
        self.Nonstationary = Nonstationary
        self.noObjs = self.arms["objectives"] 
        self.noArms = len(self.arms_distribution(0))

        self.arms_breakpoints   = []
        
        if(self.Nonstationary):
            for point in self.arms["points"]:
                self.arms_breakpoints.append(point["t1"] )
            self.arms_breakpoints.remove(0)
        
    def get_arms_mean(self,t = 0):
        means = []
        for arm in self.arms_distribution(t):
            means.append(arm.means)
        return means

    def arms_distribution(self,t=0):
        if(self.Nonstationary):
            for point in self.arms["points"]:
                if( point["t1"] <= t and point["t2"] > t):
                    distribution = self.arms["type"]
                    return [distribution(mean) for mean in point["means"] ]
        else:
            distribution = self.arms["type"]
            return [distribution(mean) for mean in self.arms["points"][0]["means"] ]
                
    def get_noArms(self,t = 0):
        self.noArms = len(self.arms_distribution(t))
        return self.noArms
    
    def get_noObjs(self,t = 0):
        return self.noObjs
    
    def pull_arm(self,noOfArm,t = 0):
        reward =  self.arms_distribution(t)[noOfArm].draw()
        return reward