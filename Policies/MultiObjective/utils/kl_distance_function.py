import numpy as np
class kl_distance_function:
    def __init__(self, epsilon =  0.00001,  alpha = 0.001):
        self.alpha = alpha
        self.epsilon = epsilon
        
    def standardization(self, p, q):
        """ Epsilon is used here to avoid conditional code for
        checking that neither P nor Q is equal to 0. """
        P = np.array(p)
        Q = np.array(q)
        P[P<self.epsilon] = self.epsilon
        Q[Q<self.epsilon] = self.epsilon
        P[P>1-self.epsilon] = 1-self.epsilon
        Q[Q>1-self.epsilon] = 1-self.epsilon
        return P,Q 
        
    def KL_bernoulli(self, p, q):
        P,Q = self.standardization(p, q)
        # KL divergence formula
        divergence = (P * np.log(P/Q) + (1-P) * np.log((1-P)/(1-Q)))
        return divergence       
    def derivative_KL(self, p, q): 
        P,Q = self.standardization(p, q)
        # derivative of KL formula
        divergence = (P/Q) + ((1-P)/(1-Q))
        return divergence  
        
    def compute_max_value(self,p,d):
        q0 = p.copy()
        q  = p.copy()
        for i in range(1000):
            q0 = q0 +  self.alpha * (self.derivative_KL(p,q0)) #derivative of KL formula
            d_q_p = self.KL_bernoulli(p,q0)
            if(np.any(d_q_p) <= d):
                q[d_q_p <= d]  = q0[d_q_p <= d] 
                q[q > 1.0] = 1.0
            else:
                break
        return q
        