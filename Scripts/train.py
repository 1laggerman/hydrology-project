
from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_run
from pathlib import Path

start_run(Path('configs/LSTM.yml'), gpu=0)  


