from utils.general import get_last_run
from utils.configs import get_last_run_config
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
from pathlib import Path

config = Config(Path('RT_flood/check_loss_config.yaml'))

print(get_last_run_config(config)._cfg)
