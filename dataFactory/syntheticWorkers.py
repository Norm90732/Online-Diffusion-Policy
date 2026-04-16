from omegaconf import DictConfig
import gymnasium as gym 

from ray import remote 


class SucessWorker():
    def __init__(self,cfg:DictConfig,taskName:str):
        
        self.taskname:str = taskName
        self.rewardMode:str = cfg.factory.baseEnvironment.rewardType
        self.obsMode: str = cfg.factory.baseEnvironment.obsMode

        self.env = gym.make()
    
    
    
    