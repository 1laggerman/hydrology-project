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

# build_basins_config('../data/Caravan', 'configs')
build_attr_config('../data/Caravan', 'configs')
# data_conf = yaml.safe_load(Path('configs', 'basins_1.yaml').read_text())

# print(data_conf)

# foo = {
#     'name': 'foo',
#     'my_list': [
#         {'foo': 'test', 'bar': 'test2'},
#         {'foo': 'test3', 'bar': 'test4'}],
#     'hello': 'world',
# }

# test_yaml()


# run from "for work" directory?

# print(Path("hydrology-project/configs/LSTM.yml").absolute())
# cfg = Config(Path("configs/LSTM.yml"))
# start_run(Path('configs/LSTM.yml'), gpu=0)
# model = get_model(cfg=cfg)

# print(model.state_dict())










# from neuralhydrology.neuralhydrology.nh_run import start_run

# start_run(config_file=Path("configs/LSTM.yml"), gpu=0)

# from neuralhydrology.nh_run import start_run, eval_run


# model = LSTM(4, 256)

# myStateDict = torch.load('runs/LSTM_test.pt', weights_only=False)
# print(myStateDict.keys())

# theirStateDict = torch.load('runs/all_data_caravan_one_layer_1411_165612/model_epoch001.pt', weights_only=False)
# print(theirStateDict.keys())

# print(theirStateDict['embedding_net.statics_embedding.net.0.bias'])
# print(myStateDict['embedding_net.bias'])

# dictmap = utils.map_state_dict(myStateDict, theirStateDict)


# print(dictmap['src_to_tgt'])
# print(dictmap['tgt_to_src'])


# print(theirStateDict['embedding_net.statics_embedding.net.0.bias'])
# print(myStateDict['embedding_net.bias'])


# model.load_state_dict(myStateDict)
# model.load_state_dict(theirStateDict)

# print('success')

# my_items = myStateDict.items()
# their_items = theirStateDict.items()

# my_items_array = []
# for key, value in my_items:
#     my_items_array.append([key, value.shape])

# their_items_array = []
# for key, value in their_items:
#     their_items_array.append([key, value.shape])

# for i in range(len(my_items)):
#     print(my_items_array[i][0], ': ', my_items_array[i][1], ', ', their_items_array[i][1])

# for key, value in their_items:
#     print(key, ': ', value.shape)
    # print(my_items[i][0], ': ', my_items[i][1].shape, ', ', their_items[i][1].shape)
# print(ret)

# if __name__ == "__main__":
#     start_run(Path('configs/test.yaml'), gpu=-1)

# eval_run(Path('runs/'), 'test')



# utils.build_meta('data\Caravan', 'configs\meta.yaml')

# df = pd.read_csv('data/Caravan/timeseries/csv/camelsaus/camelsaus_G8200045.csv')
# X, y = utils.series_to_samples(df, 4)

# print(X.shape)
# print(y.shape)
# print(type(y[7895]))
# print(type(pd.NA))
# print(y.shape)

# if pd.isna(df.iloc[7895, -1]):
#     print("yes")

# X = np.ndarray((0, 4))

# new_X = np.concatenate((X, np.array([[1, 2, 3, 4]])), axis=0)

# new_X = np.concatenate((new_X, np.array([[1, 2, 3, 4]])), axis=0)

# print(X)
# print(X.shape)
# print(new_X)
# print(new_X.shape)

# run_cfg = utils.load_config('code/configs')

# utils.load_Data(run_cfg)

# model = utils.get_model(run_cfg)

# print(type(model))


# print(run_cfg)
# save_config('code/results', 'run', run_cfg)


# data_conf = yaml.safe_load(Path('code/configs/Data.yaml').read_text())
# model_conf = yaml.safe_load(Path('code/configs/Model.yaml').read_text())

# # print(data_conf)
# # print(model_conf)

# run_conf = {}
# run_conf.update(data_conf)
# run_conf.update(model_conf)

# file_name = get_free_name('code/results', 'run', '.yaml')
# yaml.safe_dump(run_conf, Path(f'code/results/{file_name}').open('w'), sort_keys=False)
