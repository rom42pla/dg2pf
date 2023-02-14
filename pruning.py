import math
from typing import Union, Optional, List

import numpy as np
import torch
from torch import nn

from utils import get_flattened_weights, to_sparse, get_model_device


def get_pruned_perc(
        model: nn.Module,
        frozen_layers: Optional[Union[str, List[str]]] = None,
):
    q = 0
    tot = 0
    for parameter_name, parameter_group in model.named_parameters():
        # if (not parameter_name.endswith("weight")) \
        #         or (frozen_layers and parameter_name in frozen_layers):
        if (frozen_layers and parameter_name in frozen_layers):
            continue
        mask = torch.abs(parameter_group.data) == 0
        q += torch.sum(mask).item()
        tot += torch.numel(parameter_group)
    return q / tot


def get_unique_weights(copied_model):
    unique_arr = np.array([])
    for name, W in copied_model.named_parameters():
        if 'weight' in name:
            unique_arr = np.unique(np.concatenate([unique_arr, np.unique(W.data.cpu().detach().numpy().flatten())]))
    return unique_arr


def prune(
        model: nn.Module,
        pruning_percent: float,
        frozen_layers: Optional[Union[str, List[str]]] = None,
        layerwise: bool = False,
):
    assert 0 < pruning_percent < 1
    assert isinstance(layerwise, bool)

    if not layerwise:
        # finds the left and right margin for the pruning
        weights = get_flattened_weights(model=model, frozen_layers=frozen_layers)
        left_margin = np.percentile(weights, ((1 - pruning_percent) / 2) * 100)
        right_margin = np.percentile(weights, (pruning_percent + (1 - pruning_percent) / 2) * 100)
        assert right_margin >= left_margin

        # prunes the weights
        for parameter_name, parameter_group in model.named_parameters():
            # do not prune biases and frozen layers
            # if (not parameter_name.endswith("weight")) \
            #         or (frozen_layers and parameter_name in frozen_layers):
            if (frozen_layers and parameter_name in frozen_layers):
                continue
            mask = (parameter_group.data < right_margin) & \
                   (parameter_group.data > left_margin)
            parameter_group.data[mask] = 0
            assert ((parameter_group.data < right_margin) &
                    (parameter_group.data > left_margin) &
                    (parameter_group.data != 0)) \
                       .nonzero().numel() == 0
    else:
        for parameter_name, parameter_group in model.named_parameters():
            # do not prune biases and frozen layers
            # if (not parameter_name.endswith("weight")) \
            #         or (frozen_layers and parameter_name in frozen_layers):
            if (frozen_layers and parameter_name in frozen_layers):
                continue
            # finds the left and right margin for the pruning
            weights = parameter_group.data.flatten().detach().cpu()
            left_margin = np.percentile(weights, ((1 - pruning_percent) / 2) * 100)
            right_margin = np.percentile(weights, (pruning_percent + (1 - pruning_percent) / 2) * 100)
            assert right_margin >= left_margin

            # prunes the weights
            mask = (parameter_group.data < right_margin) & \
                   (parameter_group.data > left_margin)
            parameter_group.data[mask] = 0
            assert ((parameter_group.data < right_margin) &
                    (parameter_group.data > left_margin) &
                    (parameter_group.data != 0)) \
                       .nonzero().numel() == 0
