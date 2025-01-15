
from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_training
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
from pathlib import Path
from utils.configs import create_run_folder

gpu = 0
config = Config(Path('configs/working_confs/base.yaml'))

# check if a GPU has been specified as command line argument. If yes, overwrite config
if gpu is not None and gpu >= 0:
    config.device = f"cuda:{gpu}"
if gpu is not None and gpu < 0:
    config.device = "cpu"

if config.run_dir is None:
    config.run_dir = Path('runs')

create_run_folder(config)


