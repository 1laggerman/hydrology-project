
from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_training, start_run
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
from nhWrap.neuralhydrology.neuralhydrology.training.basetrainer import BaseTrainer
from pathlib import Path
from utils.configs import create_run_folder

gpu = 0
config = Config(Path('configs/working_confs/base.yaml'))

# check if a GPU has been specified as command line argument. If yes, overwrite config
if gpu is not None and gpu >= 0:
    config.device = f"cuda:{gpu}"
if gpu is not None and gpu < 0:
    config.device = "cpu"

if config.run_dir is None:
    config.run_dir = Path('runs')

# create a run folder with saved config of this run
create_run_folder(config)

# # start training
if config.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:
    trainer = BaseTrainer(cfg=config)
else:
    raise ValueError(f"Unknown head {config.head}.")


trainer.initialize_training()

# get epoch 0 loss
if (trainer.validator is not None):
    trainer.validator.evaluate(epoch=0,
                            save_results=trainer.cfg.save_validation_results,
                            save_all_output=trainer.cfg.save_all_output,
                            metrics=trainer.cfg.metrics,
                            model=trainer.model,
                            experiment_logger=trainer.experiment_logger.valid())

    valid_metrics = trainer.experiment_logger.summarise()
    print_msg = f"Epoch 0 average validation loss: {valid_metrics['avg_total_loss']:.5f}"
    print(print_msg)

# trainer.train_and_validate()


