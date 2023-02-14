from typing import Optional, Union, List

import torch
from torch import nn

from utils import get_model_device, weight_clamp, to_sparse


def quantization_fn(
        x: torch.Tensor,
        r: float,
        quantization_bits: int,
        range_clip: Union[int, float]):
    assert 0 < r <= 1
    assert isinstance(quantization_bits, int) and quantization_bits >= 1
    assert range_clip > 0
    range_clip += range_clip / ((2 ** (quantization_bits)) - 1)
    if r == 1:
        j = torch.exp2(torch.tensor(quantization_bits - 1)) / range_clip
        return (torch.ceil(x * j) / j) - \
            (1 / (torch.exp2(torch.tensor(quantization_bits)) / range_clip))
    else:
        j = torch.exp2(torch.tensor(quantization_bits)) / range_clip
        pi = torch.pi
        return (x - ((torch.arctan(
            -((r * torch.sin(j * pi * x)) / (
                    1 - (r * torch.cos(j * pi * x)))))) / (
                             pi * j * (1 / 2))))


def weight_quantize(
        model: nn.Module,
        r: float,
        bits: int,
        range_clip: Union[int, float],
        frozen_layers: Optional[Union[str, List[str]]] = None,
):
    for parameter_name, parameter_group in model.named_parameters():
        # do not prune biases and frozen layers
        # if (not parameter_name.endswith("weight")) \
        #         or (frozen_layers and parameter_name in frozen_layers):
        if (frozen_layers and parameter_name in frozen_layers):
            continue
        mask = torch.abs(parameter_group.data) > 1e-7
        parameter_group.data[~mask] = 0
        parameter_group.data[mask] = quantization_fn(parameter_group.data[mask], r, bits, range_clip)


def get_quantized_weight_model(
        model: nn.Module,
        quantized_model: nn.Module,
        quantization_bits: int,
        range_clip: Union[int, float],
        frozen_layers: Optional[Union[str, List[str]]] = None,
):
    quantized_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        weight_clamp(model=quantized_model, range_clip=range_clip, frozen_layers=frozen_layers)
        weight_quantize(quantized_model, r=1, bits=quantization_bits, range_clip=range_clip,
                        frozen_layers=frozen_layers)
        weight_clamp(model=quantized_model, range_clip=range_clip, frozen_layers=frozen_layers)
    quantized_model.eval()

    check_if_quantization_works(model=quantized_model,
                                quantization_bits=quantization_bits,
                                frozen_layers=frozen_layers)
    return quantized_model


def check_if_quantization_works(model: nn.Module,
                                quantization_bits: int,
                                frozen_layers: Optional[Union[str, List[str]]] = None, ):
    assert isinstance(quantization_bits, int) and quantization_bits >= 1
    device = get_model_device(model)
    model.cpu()
    for parameter_name, parameter_group in model.named_parameters():
        # do not quantize biases and frozen layers
        # if (not parameter_name.endswith("weight")) \
        #         or (frozen_layers and parameter_name in frozen_layers):
        if (frozen_layers and parameter_name in frozen_layers):
            continue
        unique_values = parameter_group.data.detach().unique(sorted=False)
        if unique_values.numel() - 1 > 2 ** quantization_bits:
            print(unique_values)
            raise OverflowError(f"Error during quantization. There are {unique_values.numel() - 1} unique values "
                                f"instead of leq than {2 ** quantization_bits}")
            # print("range:", -range_clip, range_clip)
            # print("unique values:", uniques_array.shape, "max representable values:", 2 ** quantization_bit)
            # print(uniques_array)
            # coef = quantization_bit / range_clip
            # j = torch.exp2(torch.tensor(coef - 1))
            # # print("actual", W.data)
            # # print("quantized", (torch.ceil(W.data * j) / j) - (1 / torch.exp2(torch.tensor(coef))))
            # print("quantizationerror:",
            #       torch.mean(parameter_group.data - (torch.ceil(parameter_group.data * j) / j) - (1 / torch.exp2(torch.tensor(coef)))))
            # step = (range_clip / torch.exp2(torch.tensor(quantization_bit - 1)))
            # print(step)
            # print([step * a + -range_clip for a in range(2 ** quantization_bit)])
    # print("unique values: ",uniquw_arr.shape[0], " -> ",uniquw_arr)
    model.to(device)


class Multiply(object):

    def __init__(self, input_scaler: float, input_shifter: float, bits: float, range_c: float, do_quantization=True):
        self.input_scaler = input_scaler
        self.bits = bits
        self.range_clip = range_c
        self.input_shifter = input_shifter
        self.do_quantization = do_quantization

    def __call__(self, sample):
        return quantization_fn((sample + self.input_shifter) * self.input_scaler, 1 if self.do_quantization else 0,
                               self.bits,
                               self.range_clip)
