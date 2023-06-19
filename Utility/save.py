import numpy as np
import os
from datetime import datetime
import shutil


def init_save(config):
    if(config.save):
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        
        save_name = config.save_name
        if(config.save_name == "auto"):
            now = datetime.now()
            save_name  = now.strftime("%Y_%m_%d_%H_%M_%S")
            
        save_path = config.save_dir + save_name + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        if(config.save_config):        
            shutil.copy2(config.save_config, save_path)
        
        return save_path
        
    else:
        
        return None
    
    
def save_observations(config, save_path, observations, obs_pulled_arms):
    if(config.save_observations == True):
        np.save(save_path+ "observations.npy", observations) # save
        np.save(save_path+ "obs_pulled_arms.npy", obs_pulled_arms) # save
    if(config.type == "real-yahoo"):
        if(os.path.exists("./Datasets/yahoo_indexes.npy")==0):
            np.save( "./Datasets/yahoo_indexes.npy",config.environment.indexes)