import numpy as np

from Utility import graph #,JCAS_graph
from Utility.save import init_save, save_observations

#run = "JCAS"
#run = "real"
run = "synthetic"


if run ==  "JCAS":
    from Expriments.JCAS._newExpriment import config_JCAS as config
    observations = np.load("./Expriments/JCAS/_newExpriment/observations.npy")
    obs_pulled_arms = np.load("./Expriments/JCAS/_newExpriment/obs_pulled_arms.npy")

    config.save_dir = "./Expriments/JCAS/"
    #config.save_name = "JCAS"
    config.save_plots = True
    config.save_config = False
    config.save_observations = False

    save_path = init_save(config)
    JCAS_graph.cumulative_pareto_regrets(config,obs_pulled_arms,save_path)
    JCAS_graph.performances(config,observations,save_path)

elif run ==  "real":
    from Expriments.real._newExpriment import config_yahoo as config
    observations = np.load("./Expriments/real/_newExpriment/observations.npy")
    obs_pulled_arms = np.load("./Expriments/real/_newExpriment/obs_pulled_arms.npy")
    
    config.save_dir = "./Expriments/real/"
    #config.save_name = "real"
    config.save_plots = True
    config.save_config = False
    config.save_observations = False

    save_path = init_save(config)
    graph.cumulative_pareto_regrets(config,obs_pulled_arms,save_path)
    graph.plot_arms_means(config,save_path)
    
elif run ==  "synthetic":
    from Expriments.synthetic._newExpriment import config_synthetic as config
    observations = np.load("./Expriments/synthetic/_newExpriment/observations.npy")
    obs_pulled_arms = np.load("./Expriments/synthetic/_newExpriment/obs_pulled_arms.npy")

    config.save_dir = "./Expriments/synthetic/"
    #config.save_name = "synthetic"
    config.save_plots = True
    config.save_config = False
    config.save_observations = False
    
    save_path = init_save(config)
    graph.cumulative_pareto_regrets(config,obs_pulled_arms,save_path)
    graph.plot_arms_means(config,save_path)