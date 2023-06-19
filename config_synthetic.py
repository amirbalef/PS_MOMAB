from Arms.MultiObjectiveBernoulliArm import MultiObjectiveBernoulliArm
from Environment.MultiObjectiveEnvironment import MultiObjectiveEnvironment

from Policies.MultiObjective.Pareto_UCB1 import Pareto_UCB1
from Policies.MultiObjective.Pareto_KL_UCB import Pareto_KL_UCB

from ChangePointDetector.RBOCPD import RBOCPD
from ChangePointDetector.MultiChangePointDetectors import Multi_CDs

from Policies.MultiObjective.CD_Pareto_KL_UCB import CD_Pareto_KL_UCB
from Policies.MultiObjective.CD_Pareto_UCB1 import CD_Pareto_UCB1


from Policies.MultiObjective.SlidingWindow_Pareto_klUCB import SlidingWindow_Pareto_klUCB
from Policies.MultiObjective.SlidingWindow_Pareto_UCB import SlidingWindow_Pareto_UCB
from Policies.MultiObjective.Discounted_Pareto_UCB import Discounted_Pareto_UCB



import numpy as np
np.random.seed(0)

type = "synthetic"
Rounds    = 100
T = 1500

arms = {}
arms["non-stationary"] = True
arms["numbers"] = 4
arms["objectives"] = 3
arms["type"] = MultiObjectiveBernoulliArm
arms_point_1 = {"t1":0*T/5,  "t2": 2*T/5, "means":[[0.50,0.60,0.45],     [0.20,0.50,0.20],    [0.30,0.30,0.30],    [0.00,0.10,0.40]]}
arms_point_2 = {"t1":2*T/5,  "t2": 3*T/5, "means":[[0.20,0.20,0.05],     [0.00,0.15,0.40],    [0.60,0.30,0.10],    [0.80,0.40,0.30]]}
arms_point_3 = {"t1":3*T/5,  "t2":7*T/10, "means":[[0.40,0.60,0.20],     [0.90,0.90,0.30],    [0.50,0.70,0.40],    [0.15,0.40,0.35]]}
arms_point_4 = {"t1":7*T/10,  "t2":4*T/5, "means":[[0.15,0.00,0.00],     [0.05,0.05,0.05],    [0.00,0.10,0.10],    [0.25,0.40,0.40]]}
arms_point_5 = {"t1":4*T/5,  "t2": 5*T/5, "means":[[0.30,0.00,0.30],     [0.05,0.05,0.05],    [0.20,0.30,0.35],    [0.10,0.20,0.10]]}
arms["points"] = [arms_point_1,arms_point_2,arms_point_3,arms_point_4,arms_point_5]

environment = MultiObjectiveEnvironment(arms,T,arms["non-stationary"])


alg_0 = Pareto_UCB1(environment.noObjs ,environment.noArms)
alg_1 = Pareto_KL_UCB(environment.noObjs ,environment.noArms)

alg_2 = Discounted_Pareto_UCB(environment.noObjs ,environment.noArms, 0.9)
alg_3 = SlidingWindow_Pareto_UCB(environment.noObjs ,environment.noArms, environment.time_horizon//10)

CDs = Multi_CDs(RBOCPD,T,environment.noObjs ,environment.noArms)

alg_4 = CD_Pareto_UCB1(CDs,environment.noObjs ,environment.noArms,steps_to_explore = environment.time_horizon)
alg_5 = CD_Pareto_KL_UCB(CDs,environment.noObjs ,environment.noArms,steps_to_explore = environment.time_horizon)

alg_6 = SlidingWindow_Pareto_klUCB(environment.noObjs ,environment.noArms, environment.time_horizon//10)

algorthims  = [alg_6, alg_5, alg_4, alg_3, alg_2, alg_1, alg_0 ]

#Save
save = True
save_dir = "./Expriments/synthetic/"
save_name = "_newExpriment"
save_plots = True
save_config = "./config_synthetic.py"
save_observations = True

