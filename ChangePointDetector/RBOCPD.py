import numpy as np

#Inspired from https://github.com/Ralami1859/Restarted-BOCPD
class RBOCPD:
    def __init__(self,Horizon,min_gap = 0.1):
        self.alphas = np.array([1.0])
        self.betas = np.array([1.0])
        self.ForecasterDistribution = np.array([1.0])
        self.PseudoDist = np.array([1.0])
        self.ChangePointEstimation = np.array([])
        self.like1 = min_gap
        self.Restart = 1        
        self.gamma = 1.0/(Horizon) # Switching Rate 
        self.t = 0
    def updateForecasterDistribution_m(self, reward):
        if reward == 1:
            likelihood = np.divide(self.alphas, self.alphas + self.betas)
        else:
            likelihood = np.divide(self.betas, self.alphas + self.betas)
        Pseudo_w0 = self.gamma*self.like1*np.sum(self.PseudoDist) # Using the simple prior
        self.PseudoDist = self.like1*self.PseudoDist
        ForecasterDistribution0 = Pseudo_w0 # Creating new Forecaster
        self.ForecasterDistribution = (1-self.gamma)*likelihood*self.ForecasterDistribution # update the previous forecasters
        self.ForecasterDistribution = np.append(self.ForecasterDistribution,ForecasterDistribution0) # Including the new forecaseter into the previons ones
        self.ForecasterDistribution =  self.ForecasterDistribution/np.sum( self.ForecasterDistribution) # Normalization for numerical purposes
        self.PseudoDist = np.append(self.PseudoDist,Pseudo_w0)
        self.PseudoDist = self.PseudoDist/np.sum(self.PseudoDist) # Normalization for numerical purposes

    def updateForecasterDistribution(self, reward):
        if reward == 1:
            likelihood = np.divide(self.alphas, self.alphas + self.betas)
        else:
            likelihood = np.divide(self.betas, self.alphas + self.betas)
        ForecasterDistribution0 = self.gamma*np.dot(likelihood, np.transpose(self.ForecasterDistribution)) # Creating new Forecaster 
        self.ForecasterDistribution = (1.0-self.gamma)*likelihood*self.ForecasterDistribution # update the previous forecasters 
        self.ForecasterDistribution = np.append(self.ForecasterDistribution,ForecasterDistribution0) # Including the new forecaseter into the previons ones
        self.ForecasterDistribution = self.ForecasterDistribution/np.sum(self.ForecasterDistribution) # Normalization for numerical purposes
    
    def updateLaplacePrediction(self,reward):
        self.alphas[:] += reward
        self.betas[:] += (1-reward)
        self.alphas = np.append(self.alphas,1) # Creating new Forecaster
        self.betas = np.append(self.betas,1) # Creating new Forecaster

    def update(self,reward):
        EstimatedBestExpert = np.argmax(self.ForecasterDistribution) #Change-point estimation
        # Restart precedure
        if not(EstimatedBestExpert == 0):
            # Reinitialization
            self.alphas = np.array([1.0])
            self.betas = np.array([1.0])
            self.ForecasterDistribution = np.array([1.0])
            self.Restart = self.t+1
  
        self.ChangePointEstimation = np.append(self.ChangePointEstimation,self.Restart+1)
        self.updateForecasterDistribution_m(reward)
        self.updateLaplacePrediction(reward) #Update the laplace predictor
        self.t = self.t +1
        return EstimatedBestExpert
        
    def info_str(self,latex=0):
            return  f"RBOCPD"
