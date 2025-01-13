import torch
import pandas as pd
import os
import numpy as np
import yaml
from pathlib import Path
from models.LSTM import LSTM

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

    free_name = file_name
    i = 1
    while os.path.exists(os.path.join(path, free_name)):
        free_name = f'{name}_{i}{extension}'
        i += 1
    
    return free_name


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
    
