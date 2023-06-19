import numpy as np
import tqdm as tq
from joblib import Parallel, delayed
import config_synthetic as config
from Utility.run import multiObjectiveExpriments
from Utility.save import init_save, save_observations


save_path = init_save(config)


observations = []
obs_pulled_arms = []
for alg in tq.tqdm(config.algorthims):
    obs, pulled_arms = zip(*Parallel(n_jobs=24, verbose=3)(delayed(multiObjectiveExpriments)(alg,config.environment,config.T) for i in range(config.Rounds)))
    observations.append(np.moveaxis(np.array(obs), 0, 1))
    obs_pulled_arms.append(pulled_arms)

save_observations(config, save_path, observations, obs_pulled_arms)