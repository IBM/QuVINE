from omegaconf import OmegaConf
from pathlib import Path

def save_config(cfg, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
