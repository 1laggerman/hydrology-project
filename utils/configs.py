from utils.general import *
import os
import pandas as pd
import yaml

def build_basins_config(src_path, dst_path):

    root_dir = os.path.join(src_path, 'timeseries', 'csv')
    basins = {}

    for directory in os.listdir(root_dir):
        basins[directory] = []
        for file in os.listdir(os.path.join(root_dir, directory)):
            if file.endswith(".csv"):
                basins[directory].append(file[:-4])

    save_config(dst_path, 'basins', basins)

def build_attr_config(src_path, dst_path):

    attr = {'ALL_KEYS': {'per_gauge': ['gauge_id', 'gauge_name', 'country'], 'per_sample': ['date']}, 'STATIC_ATTRIBUTES': {}, 'DYNAMIC_ATTRIBUTES': {}}

    dynamic_path = os.path.join(src_path, 'timeseries', 'csv')
    static_path = os.path.join(src_path, 'attributes')


    sample_path = os.path.join(static_path, 'camels')
    for file in os.listdir(sample_path):
        if file.endswith(".csv"):
            parts = file.split('_')
            attr['STATIC_ATTRIBUTES'][parts[-2]] = {'sample_path': os.path.join('attributes', 'camels', file), 'SIZE': 0, 'KEYS': [], 'names': []}

    sample_path = os.path.join(dynamic_path, 'camels')
    file = os.listdir(sample_path)[0]
    if file.endswith(".csv"):
        parts = file.split('_')
        attr['DYNAMIC_ATTRIBUTES'] = {'sample_path': os.path.join('timeseries', 'csv', 'camels', file), 'SIZE': 0, 'KEYS': [], 'names': []}
        

    # files, _ = get_files(os.path.join(src_path, 'timeseries', 'csv', 'camels'), extension='.csv')
    # sampleBaseFile = files[0]

    # files, _ = get_files(os.path.join(src_path, 'attributes', 'camels'), extension='.csv')

    # attr = {
    #     'BASE_ATTRIBUTES': {
    #         'sample_path': sampleBaseFile,
    #         'SIZE': 0,
    #         'KEYS': ['date'],
    #         'names': []
    #     },
    #     'CARAVAN_ATTRIBUTES': {
    #         'sample_path': next((s for s in files if 'caravan' in s)),
    #         'SIZE': 0,
    #         'KEYS': ['gauge_id'],
    #         'names': [],
    #     },
    #     'HYDROATLAS_ATTRIBUTES': {
    #         'sample_path': next((s for s in files if 'hydroatlas' in s)),
    #         'SIZE': 0,
    #         'KEYS': ['gauge_id'],
    #         'names': [],
    #     },
    #     'OTHER_ATTRIBUTES': {
    #         'sample_path': next((s for s in files if 'other' in s)),
    #         'SIZE': 0,
    #         'KEYS': ['gauge_id', 'gauge_name', 'country'],
    #         'names': [],
    #     }
    # }

    for key, value in attr['STATIC_ATTRIBUTES'].items():

        df = pd.read_csv(os.path.join(src_path, value['sample_path']))

        names = df.columns.to_list()
        keys = []

        for key in attr['ALL_KEYS']['per_gauge']:
            try: 
                names.remove(key)
                keys.append(key)
            except ValueError:
                pass

        value['KEYS'] = keys
        value['SIZE'] = len(names)
        value['names'] = names

    key = 'DYNAMIC_ATTRIBUTES'
    value = attr[key]

    df = pd.read_csv(os.path.join(src_path, value['sample_path']))

    names = df.columns.to_list()
    keys = []

    for key in attr['ALL_KEYS']['per_sample']:
        try: 
            names.remove(key)
            keys.append(key)
        except ValueError:
            pass

    value['KEYS'] = keys
    value['SIZE'] = len(names)
    value['names'] = names


    # print(attr)
    save_config(dst_path, 'attributes', attr)

    return attr

def load_config(path):

    data_conf = yaml.safe_load(Path(path, 'Data.yaml').read_text())


    model_conf = yaml.safe_load(Path(path, 'Model.yaml').read_text())

    run_conf = {}
    run_conf.update(data_conf)
    run_conf.update(model_conf)

    return run_conf

class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)
    
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if (self.indent == 0):
            self.stream.write('\n')

def save_config(path, name, config):

    file_name = get_free_name(path, name, '.yaml')
    yaml.dump(config, Path(path, file_name).open('w'), Dumper=MyDumper, default_flow_style=False)