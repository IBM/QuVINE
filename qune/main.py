import os 
import sys 
import logging 
from pathlib import Path 

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from qune.pipeline import Pipeline
from qune.utils.seed import set_global_seed 
from qune.utils.logging import setup_logging 
from qune.utils.io import save_config 

@hydra.main(
    version_base="1.3", 
    config_path="../configs/", 
    config_name="config"
)

def main(cfg: DictConfig) -> None: 
    """
    Hydra entrypoint for qune experiments
    """
    
    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    
    run_dir = Path(os.getcwd()) #hydra changes cwd
    orig_cwd = Path(get_original_cwd()) 
    
    cfg.runtime = {
        "run_dir": str(run_dir),
        "project_root": str(orig_cwd)
    }
    
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    
    setup_logging(run_dir) 
    log = logging.getLogger(__name__)
    log.info("Starting qune run")
    log.info("Run Directory: %s", run_dir)
    log.info("Project Root: %s", orig_cwd)
    
    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    
    if "seed" in cfg: 
        set_global_seed(cfg.seed)
        log.info("Global seed set to %d", cfg.seed)
        
    # ------------------------------------------------------------------
    # Resolved Config
    # ------------------------------------------------------------------
    
    save_config(cfg, run_dir/"config.yaml")
    log.debug("Resolved config: \n%s", OmegaConf.to_yaml(cfg))
    
    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    try:
        pipeline = Pipeline(cfg)
        pipeline.run()

    except Exception as e:
        log.exception("Experiment failed with error")
        raise e

    log.info("Run completed successfully")
    
if __name__ == "__main__":
    main()