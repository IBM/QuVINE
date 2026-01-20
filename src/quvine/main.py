import os 
import sys 
import logging 
from pathlib import Path 

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from quvine.pipeline import Pipeline
from quvine.utils.seed import set_global_seed 
from quvine.utils.logging import setup_logging 
from quvine.utils.io import save_config 

@hydra.main(
    
    config_path=None, 
    config_name=None,
    version_base="1.3"
    
)

def main(cfg: DictConfig) -> None: 
    """
    Hydra entrypoint for quvine experiments
    """
    
    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    
    run_dir = Path(os.getcwd()) #hydra changes cwd
    orig_cwd = Path(get_original_cwd()) 
    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    
    setup_logging(run_dir) 
    log = logging.getLogger(__name__)
    log.info("Starting quvine run")
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