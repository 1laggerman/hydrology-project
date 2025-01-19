import os
import pandas as pd
import yaml
import torch


from utils.configs import build_basins_config, build_attr_config
# from utils.configs import test_yaml
# from utils.configs import load_config

from models.LSTM import LSTM
import numpy as np
from pathlib import Path
import sys

from nhWrap.neuralhydrology.neuralhydrology.nh_run import Config
from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_run
from nhWrap.neuralhydrology.neuralhydrology.modelzoo import get_model
import xarray

df = xarray.open_dataset(Path('../data/Caravan/timeseries/netcdf/camels/camels_01022500.nc')).to_dataframe()