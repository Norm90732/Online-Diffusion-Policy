from omegaconf import DictConfig
import gymnasium as gym 
import mani_skill.trajectory.utils as trajectory_utils
import h5py
import json 
import ray  
from ray.util.queue import Queue, Full 

#returns h5, json for reading in syntehtic worker 
def savedTrajectoryParser(cfg:DictConfig) -> tuple[str,str]: 
    return ("s","s")

#https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
class SyntheticWorker():
    def __init__(self,cfg:DictConfig,taskName:str,savedReplayDir:tuple[str,str]):
        self.hdf5File= savedReplayDir[0]
        self.jsonFile = savedReplayDir[1]
        self.taskname:str = taskName
        self.numEnvs = cfg.factory.baseEnvironment.numEnvironments
        self.rewardMode:str = cfg.factory.baseEnvironment.rewardType
        self.obsMode: str = cfg.factory.baseEnvironment.obsMode
    
    def _loadHDF5(self,episodeId):
        with h5py.File(self.hdf5File,"r") as f:
            trajKey = f"traj_{episodeId}"
            
            traj  = f[trajKey]
            
            actions = traj["actions"][:] #T,A 
            terminated = traj["terminated"][:]
            truncated = traj["truncated"][:]
            envState = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            
            currentDict = {
                "actions": actions,
                "terminated": terminated,
                "truncated": truncated,
                "envState": envState,
            }
            if traj.get("success") is not None:
                currentDict["success"] = traj["success"][:]
            if traj.get("fail") is not None:
                currentDict["fail"] = traj["fail"][:]
            if traj.get("obs") is not None:
                currentDict["obs"] = traj["obs"][:]

            return currentDict
           
    def _loadJSON(self):
        with open(self.jsonFile,"r") as f:
            data = json.load(f)
            return {
                "env_info": data["env_info"],
                "episodes": data["episodes"],
            }
            
    def _buildEnv(self):
        pass 
    
    