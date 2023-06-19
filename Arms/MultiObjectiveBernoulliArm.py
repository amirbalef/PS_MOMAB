import numpy as np

class MultiObjectiveBernoulliArm:
    def __init__(self, probabilities):
        self.means = probabilities  #: Mean for this Bernoulli arm
    def draw(self):
        return np.random.binomial(1, self.means)
        
        