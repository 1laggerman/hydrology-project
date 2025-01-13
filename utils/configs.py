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
                basins[directory].append(file)

    save_config(dst_path, 'basins', basins)



def build_attr_config(src_path, dst_path):

    files, _ = get_files(os.path.join(src_path, 'timeseries', 'csv', 'camels'), extension='.csv')
    sampleBaseFile = files[0]

    files, _ = get_files(os.path.join(src_path, 'attributes', 'camels'), extension='.csv')

    attr = {
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

    for key, value in attr.items():
        df = pd.read_csv(value['sample_path'])
        df.drop(value['KEYS'], axis=1, inplace=True)
        names = df.columns.to_list()
        value['names'] = names
        value['SIZE'] = len(names)

    # yaml.safe_dump(meta, open(dst_path, 'w'), sort_keys=False)
    save_config(dst_path, 'attributes', attr)

    return attr

def load_config(path):

    data_conf = yaml.safe_load(Path(path, 'Data.yaml').read_text())


    model_conf = yaml.safe_load(Path(path, 'Model.yaml').read_text())

    run_conf = {}
    run_conf.update(data_conf)
    run_conf.update(model_conf)

    return run_conf

# class IndentedDumper(yaml.Dumper):

#     def increase_indent(self, flow=False, indentless=False):
#         return super(IndentedDumper, self).increase_indent(flow, False)

# def save_config(path, name, config):
#     file_name = get_free_name(path, name, '.yaml')

    # with open(file_name, 'w') as file:
    #     yaml.dump(config, file_name, Dumper=IndentedDumper, default_flow_style=False)

    # yaml.safe_dump(config, Path(path, file_name).open('w'), sort_keys=False, Dumper=TabDumper, default_flow_style=False, indent=1, width=2)

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