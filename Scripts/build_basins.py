from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_training
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
from pathlib import Path
from utils.configs import *

selector = {
    'camels': 'all',
    'camelsaus': 'all',
    'camelsbr': 'all',
    'camelscl': 'all',
    'camelsgb': 'all',
    'hysets': 'all',
    'lamah': 'all',
}

generate_basins_txt(selector, 'configs/all_basins.yaml', 'configs/basins/all_basins.txt')