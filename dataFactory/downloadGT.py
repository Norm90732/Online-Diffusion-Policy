#https://maniskill.readthedocs.io/en/latest/user_guide/datasets/demos.html
from omegaconf import DictConfig
from pathlib import Path 
import subprocess
import sys

def downloadHDF5GT(cfg:DictConfig) -> None:
    downloadDir = cfg.factory.replaySaveDir
    Path(downloadDir).mkdir(parents=True, exist_ok=True)
    
    for env in cfg.factory.envs: 
        subprocess.run([ #pyrefly:ignore 
                sys.executable,
                "-m",
                "mani_skill.utils.download_demo",
                env,
                "-o",
                str(downloadDir),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        
        
    print(f"Completed Download saved to {str(downloadDir)}")
    
    return None 
"""
if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/config.yaml")
    cfg.factory = OmegaConf.load("configs/factory/pandaWristCam.yaml")
    cfg= DictConfig(cfg)
    downloadHDF5GT(cfg=cfg)
"""
    