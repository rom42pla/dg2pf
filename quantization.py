from typing import Union

import numpy as np
import torch
from torch import nn


class WeightClipper:

    def __init__(self, range_clip):
        self.range_clip = range_clip

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.range_clip, +self.range_clip)
            module.weight.data = w

def check_if_quantization_works(copyed_model, range_clip, quantization_bit):
    uniquw_arr = np.array([])
    for name, W in copyed_model.named_parameters():
        if 'weight' in name:
            # print(W[0])
            uniquw_arr = np.unique(np.concatenate([uniquw_arr, np.unique(W.data.cpu().detach().numpy().flatten())]))
            if uniquw_arr.shape[0] - 1 > 2 ** quantization_bit:
                print("range:", -range_clip, range_clip)
                print("unique values:", uniquw_arr.shape, "max rappresetnable values:", 2 ** quantization_bit)
                print(uniquw_arr)
                coef = quantization_bit / range_clip
                j = torch.exp2(torch.tensor(coef - 1))
                # print("actual", W.data)
                # print("quantized", (torch.ceil(W.data * j) / j) - (1 / torch.exp2(torch.tensor(coef))))
                print("quantizationerror:",
                      torch.mean(W.data - (torch.ceil(W.data * j) / j) - (1 / torch.exp2(torch.tensor(coef)))))
                step = (range_clip / torch.exp2(torch.tensor(quantization_bit - 1)))
                print(step)
                print([step * a + -range_clip for a in range(2 ** quantization_bit)])
    # print("unique values: ",uniquw_arr.shape[0], " -> ",uniquw_arr)

def weight_quantize(
        model: nn.Module,
        r: float,
        bits: int,
        range_clip: Union[int, float]
):
    with torch.no_grad():
        for parameter_name, parameter_group in model.named_parameters():
            if 'weight' in parameter_name:
                # step = range_clip / torch.exp2(torch.tensor(bits - 1))
                mask = torch.abs(parameter_group.data) > 1e-7
                # mask = torch.abs((W_.data // step) * step) > 0.00000001
                parameter_group.data[torch.logical_not(mask)] = 0

                if r == 1:
                    j_ = torch.exp2(torch.tensor(bits - 1)) / range_clip
                    parameter_group.data[mask] = (torch.ceil(parameter_group.data[mask] * j_) / j_) - \
                                                 (1 / (torch.exp2(torch.tensor(bits)) / range_clip))
                else:
                    j = torch.exp2(torch.tensor(bits)) / range_clip
                    pi = torch.pi

                    parameter_group.data[mask] = (parameter_group.data[mask] - ((torch.arctan(
                        -((r * torch.sin(j * pi * parameter_group.data[mask])) / (
                                1 - (r * torch.cos(j * pi * parameter_group.data[mask])))))) / (
                                                                                        pi * j * (1 / 2))))