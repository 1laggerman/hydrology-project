import pandas as pd
import torch
from utils import series_to_samples, get_files, clean_series
import sys
from pathlib import Path



























# from neuralhydrology.neuralhydrology.nh_run import start_run, eval_run

# files, dirs = get_files('data/Caravan/timeseries/csv')
# print(files[:10])
# df = pd.read_csv('data/Caravan/timeseries/csv/camelsaus/camelsaus_G8200045.csv')

# a, b = series_to_samples(df, 4)
# print(a.shape)



# print(df[2600:2610])
# df = clean_series(df)

# for batch_id, batch_data in df.groupby('Batch'):
#     print(batch_data.drop('Batch', axis=1))
#     if batch_id == 2:
#         break

# print(df[2600:2610])

# print(df[4267: 4280])

# df = df.dropna(axis=0)
# # print(df)
# timeseries = df.drop('date', axis=1).values.astype('float32')
# print(timeseries)
# # timeseries = df[["Passengers"]].values.astype('float32')