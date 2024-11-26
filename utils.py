import torch
import pandas as pd
import os
import numpy as np
import yaml
from pathlib import Path
from models.LSTM import LSTM


def build_meta(src_path, dst_path):
    # sampleBaseFile = os.path.join(path, 'timeseries', 'csv', 'cameles', 'camels_01022500.csv')

    files, _ = get_files(os.path.join(src_path, 'timeseries', 'csv', 'camels'), extension='.csv')
    sampleBaseFile = files[0]

    files, _ = get_files(os.path.join(src_path, 'attributes', 'camels'), extension='.csv')

    meta = {
        'BASE_ATTRIBUTES': {
            'sample_path': sampleBaseFile,
            'SIZE': 0,
            'KEYS': ['date'],
            'names': []
        },
        'CAMELS_ATTRIBUTES': {
            'sample_path': next((s for s in files if 'caravan' in s)),
            'SIZE': 0,
            'KEYS': ['gauge_id'],
            'names': [],
        },
        'HYDROATLAS_ATTRIBUTES': {
            'sample_path': next((s for s in files if 'hydroatlas' in s)),
            'SIZE': 0,
            'KEYS': ['gauge_id'],
            'names': [],
        },
        'OTHER_ATTRIBUTES': {
            'sample_path': next((s for s in files if 'other' in s)),
            'SIZE': 0,
            'KEYS': ['gauge_id', 'gauge_name', 'country'],
            'names': [],
        }
    }

    for key, value in meta.items():
        df = pd.read_csv(value['sample_path'])
        df.drop(value['KEYS'], axis=1, inplace=True)
        names = df.columns.to_list()
        value['names'] = names
        value['SIZE'] = len(names)

    yaml.safe_dump(meta, open(dst_path, 'w'), sort_keys=False)

    return meta


def clean_series(df: pd.Series):

    df = df.copy()

    df = df.dropna(axis=0)

    df['date'] = pd.to_datetime(df['date'])

    df['DateDiff'] = df['date'].diff().dt.days

    # Assign a unique group/batch ID to each series of consecutive days
    df['Batch'] = (df['DateDiff'] > 1).cumsum()

    df = df.drop('DateDiff', axis=1)
    df = df.drop('date', axis=1)

    return df


def series_to_samples(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []

    dataset = dataset.drop('date', axis=1)

    for i in range(len(dataset)-lookback):
        if pd.isna(dataset.iloc[i + lookback, -1]):
            continue
        feature = dataset[i : i + lookback]
        target = dataset.iloc[i + lookback, -1]
        X.append(feature)
        y.append(target)

    return np.array(X), np.array(y)

def build_dataset(name, lookback = 4):
    src_path = 'data/Caravan/timeseries/csv'

    # files, _ = get_files(os.path.join(src_path, name), extension='.csv')

    # X, y = np.ndarray((0, lookback, in_features)), np.ndarray((0, lookback, in_features))
    # for file in files:
    #     df = pd.read_csv(file)
    #     X, y = series_to_samples(df, lookback)
    #     print(X.shape, y.shape)

    # if data


def get_files(directory, extension='.csv'):
    """Walks through a directory tree and returns a list of all csv file's paths
    
    Args:
        directory: The root directory to start the search from
    """
    files = []
    dirs = []

    # Walk through all subdirectories of the given directory
    for root, dirs_in_dir, files_in_dir in os.walk(directory):
        for file in files_in_dir: # Check if the file is a csv file
            if file.endswith(extension):
                # If yes, add the full path to the list of files
                files.append(os.path.join(root, file))
        dirs.extend(dirs_in_dir)
        
    
    return files, dirs

def get_free_name(path, name: str, extension='.yaml'):

    file_name = f'{name}{extension}'

    # open(os.path.join(path, file_name), 'w').close()

    free_name = file_name
    i = 1
    while os.path.exists(os.path.join(path, free_name)):
        free_name = f'{name}_{i}{extension}'
        i += 1
    
    return free_name

def load_config(path):

    data_conf = yaml.safe_load(Path(path, 'Data.yaml').read_text())


    model_conf = yaml.safe_load(Path(path, 'Model.yaml').read_text())

    run_conf = {}
    run_conf.update(data_conf)
    run_conf.update(model_conf)

    return run_conf

def save_config(path, name, config):
    file_name = get_free_name(path, name, '.yaml')
    yaml.safe_dump(config, Path(path, file_name).open('w'), sort_keys=False, line_break='\r\n')

def get_model(config):

    kwargs = {}
    kwargs.update(config['Model'])

    kwargs.pop('name')

    if config['Model']['name'] == 'LSTM':
        return LSTM(**kwargs)
    
def load_Data(config):
    datasets = config['Datasets']
    print(datasets)

    for key, value in datasets.items():
        if value:
            build_dataset(key)

def map_state_dict(target_state_dict, source_state_dict):
    mapped_state_dict = {'src_to_tgt': {}, 'tgt_to_src': {}}
    for key, value in target_state_dict.items():
        names = key.split('.')
        matched = True
        for name in names:
            if not matched:
                break
            matched = False
            for skey, svalue in source_state_dict.items():
                tkeys = skey.split('.')
                if name in tkeys:
                    matched = True
                    break

        if matched and source_state_dict[skey].shape == value.shape:
            mapped_state_dict['src_to_tgt'][skey] = key
            mapped_state_dict['tgt_to_src'][key] = skey
    return mapped_state_dict
    
