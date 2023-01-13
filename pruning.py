from typing import Union

import numpy as np
import torch
from torch import nn


def get_max_abs_weight(
        model: nn.Module
) -> float:
    max_abs_weight: float = float(
        torch.max(
            torch.abs(
                torch.cat([
                    parameter_group.flatten()
                    for parameter_name, parameter_group in model.named_parameters()
                    if "weight" in parameter_name
                ])
            )
        ).detach().cpu().item()
    )
    return max_abs_weight


def get_pruned_perc(model):
    q = 0
    tot = 0
    for parameter_name, parameter_group in model.named_parameters():
        if 'weight' in parameter_name:
            mask = torch.abs(parameter_group.data) == 0

            q += torch.sum(mask).item()
            tot += torch.numel(parameter_group)
    return q / tot, q, tot

def get_unique_weights(copied_model):
    unique_arr = np.array([])
    for name, W in copied_model.named_parameters():
        if 'weight' in name:
            unique_arr = np.unique(np.concatenate([unique_arr, np.unique(W.data.cpu().detach().numpy().flatten())]))
    return unique_arr


def prune(
        model: nn.Module,
        beta: Union[int, float] = 50,
        percent: float = 0.925,
):
    assert beta > 0
    assert 0 < percent < 1
    # todo optimize
    # get percentage of pruned parameters (= 0)
    old_perc = get_pruned_perc(model)[0]

    weights = None
    for name, W in model.named_parameters():
        if 'weight' in name:
            if weights is None:
                weights = W.data.flatten()
            else:
                weights = torch.concatenate([weights, W.data.flatten()])
    margin = np.percentile(weights.detach().cpu(), percent * 100)
    abbs = torch.abs(weights)
    nw = torch.numel(weights)
    while torch.sum(abbs < margin) / nw < percent:
        margin += 0.0001
    for parameter_name, parameter_group in model.named_parameters():
        # prune weights only, not biases
        if 'weight' in parameter_name:
            # prunes weights near to the zero of the gaussian
            parameter_group.data[torch.abs(parameter_group.data) < margin] = 0
    print(get_pruned_perc(model))