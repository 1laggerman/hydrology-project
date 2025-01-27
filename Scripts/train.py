
from nhWrap.neuralhydrology.neuralhydrology.nh_run import start_training, start_run
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
from nhWrap.neuralhydrology.neuralhydrology.training.basetrainer import BaseTrainer, LOGGER
from pathlib import Path
from utils.configs import add_run_config, create_run_dir

gpu = -1
config = Config(Path('RT_flood/check_loss_config.yaml'))

# check if a GPU has been specified as command line argument. If yes, overwrite config
if gpu is not None and gpu >= 0:
    config.device = f"cuda:{gpu}"
if gpu is not None and gpu < 0:
    config.device = "cpu"

# if config.run_dir is None:
#     config.run_dir = Path('runs')

create_run_dir(config)

# # start training
if config.head.lower() in ['regression', 'gmm', 'umal', 'cmal', '']:
    trainer = BaseTrainer(cfg=config)
else:
    raise ValueError(f"Unknown head {config.head}.")


# add run config file to folder
add_run_config(trainer)

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
    # print(print_msg)
    if trainer.cfg.metrics:
        print_msg += f" -- Median validation metrics: "
        print_msg += ", ".join(f"{k}: {v:.5f}" for k, v in valid_metrics.items() if k != 'avg_total_loss')
        LOGGER.info(print_msg)

trainer.train_and_validate()


