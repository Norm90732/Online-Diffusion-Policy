from omegaconf import DictConfig
import torch 
import gymnasium as gym 
import mani_skill.trajectory.utils as trajectory_utils
import h5py
import json 
from cosmos_predict2
import numpy as np 
import ray  
from ray.util.queue import Queue, Full 
from ray.util.metrics import Histogram, Counter
from ray.experimental.collective import create_collective_group
#https://docs.ray.io/en/latest/ray-core/api/direct-transport.html 

#returns h5, json for reading in syntehtic worker 
def savedTrajectoryParser(cfg:DictConfig) -> tuple[str,str]: 
    return ("s","s")

class ActionNoise():
    def __init__(self,cfg:DictConfig):
        self.cfg = cfg.noiseStats
    def sampleNoise(self,action):
        if self.cfg.distribution == "gaussian":
            z = torch.randn_like(action)
            z = self.cfg.mean + (self.cfg.std * z)
            return z 
        
        elif self.cfg.distribution == "logitNormal":
            z = torch.randn_like(action)
            z = self.cfg.mean + (self.cfg.std * z)
            tSample = torch.sigmoid(z)
            s = self.cfg.shift
            shiftedNoise = (s * tSample) / (1.0 + (s - 1.0) * tSample)
            
            return shiftedNoise


#https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
@ray.remote(num_cpus=1,
            num_gpus=0.05,
            )
class SyntheticOfflineWorker():
    def __init__(self,cfg:DictConfig,taskName:str,savedReplayDir:tuple[str,str]):
        self.hdf5File= savedReplayDir[0]
        self.jsonFile = savedReplayDir[1]
        self.taskname:str = taskName
        self.numEnvs = cfg.factory.baseEnvironment.numEnvironments
        self.rewardMode:str = cfg.factory.baseEnvironment.rewardType
        self.obsMode: str = cfg.factory.baseEnvironment.obsMode
        self.renderMode = cfg.factory.baseEnvironment.renderMode
        self.simBackend = cfg.factory.baseEnvironment.simBackend
        self.sensorSize = cfg.factory.baseEnvironment.sensorSize
        self.timeHorizon = cfg.factory.timeHorizon
                
        self.distributution = ActionNoise(cfg)

        self.env = None 
        
        self.jsonData = self._loadJSON()
        self.episodes = self.jsonData["episodes"]
        
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
        jsonData = self.jsonData
        
        env_info = jsonData["env_info"]
        env_kwargs = dict(env_info["env_kwargs"])

        env_kwargs["num_envs"] = self.numEnvs
        env_kwargs["obs_mode"] = self.obsMode
        env_kwargs["reward_mode"] = self.rewardMode
        
        env_kwargs.update(
        dict(
            num_envs=self.numEnvs,
            obs_mode=self.obsMode,
            reward_mode=self.rewardMode,
            render_mode=self.renderMode,
            sim_backend=self.simBackend,
            sensor_configs=dict(width=self.sensorSize.width, height=self.sensorSize.height),
        )
    )
        
        
        self.env = gym.make(env_info["env_id"], **env_kwargs)
        return self.env

    def resetEnv(self, seed: int | None = None):
        if self.env is None:
            self._buildEnv()
        obs, info = self.env.reset(seed=seed) #pyrefly: ignore 
        return obs, info

    def stepEnv(self, actionBatch):
        if self.env is None:
            self._buildEnv() 
            self.env.reset()#pyrefly:ignore 
        return self.env.step(actionBatch) #pyrefly:ignore 

    def _sampleEpisodeIds(self):
        jsonData =self.jsonData
        episodes = self.episodes
        sampled = np.random.choice(len(episodes), size=self.numEnvs, replace=True)
        return [episodes[i]["episode_id"] for i in sampled]

    def _sampleTrajectoryMetadata(self):
        episodeIds = self._sampleEpisodeIds()
        sampledMetaData = []

        for episodeId in episodeIds:
            trajData = self._loadHDF5(episodeId)
            numTransitions = len(trajData["actions"])
            t = np.random.randint(0, numTransitions)

            item = {
                "episodeId": episodeId,
                "timestep": t,
                "action": trajData["actions"][t],
                "terminated": trajData["terminated"][t],
                "truncated": trajData["truncated"][t],
                "envState": trajData["envState"][t],
            }

            if "success" in trajData:
                item["success"] = trajData["success"][t]
            if "fail" in trajData:
                item["fail"] = trajData["fail"][t]
            if "obs" in trajData:
                item["obs"] = trajData["obs"][t]

            sampledMetaData.append(item)

        return sampledMetaData

    def _sampleActionBatchFromTrajectories(self):
        sampledMetaData = self._sampleTrajectoryMetadata()
        actionBatch = np.stack([item["action"] for item in sampledMetaData], axis=0) #pyrefly:ignore 
        return actionBatch, sampledMetaData

    #Generation Method 
    def _restoreEnvStateBatch(self,sampledMetaData) -> None:
        if self.env is None:
            self._buildEnv() 
            self.env.reset() #pyrefly:ignore 

        stateList = [item["envState"] for item in sampledMetaData]
        batchedState = trajectory_utils.list_of_dicts_to_dict(stateList)
        self.env.set_state_dict(batchedState) #pyrefly:ignore 
        return None 
    
    @ray.method(tensor_transport="nccl")
    def generateSyntheticBatch(self):
        actionBatch, sampledMetaData = self._sampleActionBatchFromTrajectories()
        
        self._restoreEnvStateBatch(sampledMetaData)
        
        
        noisedActions = self.distributution.sampleNoise(actionBatch)
        noisedActions = np.clip(noisedActions,-1.0,1.0) #pyrefly:ignore 
        
        nextObs, rewards, terminated, truncated, infos = self.stepEnv(noisedActions)
        
    
    
    