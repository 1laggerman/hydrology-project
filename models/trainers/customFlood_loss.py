import nhWrap
from nhWrap.neuralhydrology.neuralhydrology.training import loss
from nhWrap.neuralhydrology.neuralhydrology.training.loss import BaseLoss
from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config

import torch
from typing import Dict
import warnings

class MaskedWeightedMSELoss(BaseLoss):
    """Mean squared error loss.

    To use this loss in a forward pass, the passed `prediction` dict must contain
    the key ``y_hat``, and the `data` dict must contain ``y``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super(MaskedWeightedMSELoss, self).__init__(cfg, prediction_keys=['y_hat'], ground_truth_keys=['y'])

    def _get_loss(self, prediction: Dict[str, torch.Tensor], ground_truth: Dict[str, torch.Tensor], **kwargs):
        mask = ~torch.isnan(ground_truth['y'])
        loss = 0.5 * torch.mean(ground_truth['y'][mask] * (prediction['y_hat'][mask] - ground_truth['y'][mask])**2)
        return loss
    
def get_loss_obj(cfg: Config) -> loss.BaseLoss:
    """Get loss object, depending on the run configuration.
    
    Currently supported are 'MSE', 'NSE', 'RMSE', 'GMMLoss', 'CMALLoss', and 'UMALLoss'.
    
    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    loss.BaseLoss
        A new loss instance that implements the loss specified in the config or, if different, the loss required by the 
        head.
    """
    if cfg.loss.lower() == "mse":
        loss_obj = loss.MaskedMSELoss(cfg)
    elif cfg.loss.lower() == "nse":
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "weightednse":
        warnings.warn("'WeightedNSE loss has been removed. Use 'NSE' with 'target_loss_weights'", FutureWarning)
        loss_obj = loss.MaskedNSELoss(cfg)
    elif cfg.loss.lower() == "rmse":
        loss_obj = loss.MaskedRMSELoss(cfg)
    elif cfg.loss.lower() == "gmmloss":
        loss_obj = loss.MaskedGMMLoss(cfg)
    elif cfg.loss.lower() == "cmalloss":
        loss_obj = loss.MaskedCMALLoss(cfg)
    elif cfg.loss.lower() == "umalloss":
        loss_obj = loss.MaskedUMALLoss(cfg)
    elif cfg.loss.lower() == "wmse":
        loss_obj = MaskedWeightedMSELoss(cfg)
    else:
        raise NotImplementedError(f"{cfg.loss} not implemented or not linked in `get_loss()`")

    return loss_obj