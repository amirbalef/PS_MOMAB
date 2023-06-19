import numpy as np

class Multi_CDs:
    def __init__(self, CD_algorithm, time_horizon, noOfObjs ,noOfArms,min_gap = 0.1) :
        self.noOfArms = noOfArms
        self.noOfObjs = noOfObjs
        self.time_horizon = time_horizon
        self.min_gap = min_gap
        self.CDs = []
        self.t = 0
        for i in range(self.noOfArms):
            CD_objs = []
            for j in range(self.noOfObjs):
                CD_objs.append(CD_algorithm(self.time_horizon,self.min_gap))
            self.CDs.append(CD_objs)
                
    def update(self,arm,reward):
        detection = 0
        for j in range(self.noOfObjs):
            detection = self.CDs[arm][j].update(reward[j]) + detection
        #if(detection != 0):
        #    print( str(self.t) + ": detection")
        return detection
        
    def info_str(self,latex=0):
            return  self.CDs[0][0].info_str(latex)