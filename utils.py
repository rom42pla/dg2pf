import json
import random
from os import remove
from os.path import join, getsize
from typing import Tuple, Dict, List, Union, Optional

import numpy as np
import torch
import torchvision.transforms as T
from PIL.Image import Image
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_recall, multiclass_precision, \
    multiclass_accuracy

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def save_json(object, path: str):
    with open(path, "w") as json_file:
        json.dump(object, json_file, indent=4, sort_keys=True)


def get_model_device(model):
    return "cuda" if "cuda" in str(next(model.parameters()).device) else "cpu"


def get_classification_metrics(
        y_pred: torch.Tensor,
        y_gt: torch.Tensor,
        losses: Optional[torch.Tensor] = None):
    # y_pred, y_gt = y_pred.cpu(), y_gt.cpu()
    metrics = {
        "acc@1": multiclass_accuracy(y_pred, y_gt, num_classes=y_pred.shape[-1], top_k=1) * 100,
        "acc@5": multiclass_accuracy(y_pred, y_gt, num_classes=y_pred.shape[-1], top_k=5) * 100,
        "precision": multiclass_precision(y_pred, y_gt, num_classes=y_pred.shape[-1]) * 100,
        "recall": multiclass_recall(y_pred, y_gt, num_classes=y_pred.shape[-1]) * 100,
        "f1": multiclass_f1_score(y_pred, y_gt, num_classes=y_pred.shape[-1]) * 100,
    }
    if losses is not None:
        metrics["mean_loss"] = losses.mean()
    return metrics


def get_object_detection_metrics(
        y_pred: List[Dict[str, torch.Tensor]],
        y_gt: List[Dict[str, torch.Tensor]], ):
    metrics_fn = MeanAveragePrecision()
    metrics_fn.update(y_pred, y_gt)
    metrics = {
        k: v * 100
        for k, v in metrics_fn.compute().items()
    }
    return metrics


def coco_collate_fn(b: Tuple[Image, Dict[str, Union[List[float], int]]]):
    b = list(b)
    b[0] = T.Compose([
        T.ToTensor(),
    ])(b[0])
    b[1] = {
        "boxes": torch.as_tensor([object['bbox'] for object in b[1]]),
        "labels": torch.as_tensor([object['category_id'] for object in b[1]]),
    }
    b[1]["boxes"][:, 2] += b[1]["boxes"][:, 0]
    b[1]["boxes"][:, 3] += b[1]["boxes"][:, 1]
    return b


def get_max_abs_weight(
        model: nn.Module,
        frozen_layers: Optional[Union[str, List[str]]] = None,
) -> float:
    max_abs_weight: float = float(
        torch.max(
            torch.abs(
                torch.cat([
                    parameter_group.flatten()
                    for parameter_name, parameter_group in model.named_parameters()
                    if (not parameter_name.endswith("weight")) \
                       or (frozen_layers and parameter_name in frozen_layers)
                ])
            )
        ).detach().cpu().item()
    )
    return max_abs_weight


def get_flattened_weights(
        model: nn.Module,
        frozen_layers: Optional[Union[str, List[str]]] = None,
) -> torch.Tensor:
    weights = None
    for parameter_name, parameter_group in model.named_parameters():
        # do not prune biases and frozen layers
        if (not parameter_name.endswith("weight")) \
                or (frozen_layers and parameter_name in frozen_layers):
            continue
        if weights is None:
            weights = parameter_group.data.clone().to_dense().flatten().detach().cpu()
        else:
            weights = torch.concatenate([weights, parameter_group.data.clone().to_dense().flatten().detach().cpu()])
    return weights

def to_sparse(
        model: nn.Module,
):
    device = get_model_device(model)
    model.cpu()
    for name, module in model.named_modules():
        parameters_name = [name for name, l in module.named_parameters()]
        if "weight" in parameters_name:
            module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
        if "bias" in parameters_name:
            module.bias = torch.nn.Parameter(module.bias.data.to_sparse())
    model.to(device)

def weight_clamp(
        model,
        range_clip: Union[int, float],
        frozen_layers: Optional[Union[str, List[str]]] = None,
):
    assert range_clip > 0
    with torch.no_grad():
        for parameter_name, parameter_group in model.named_parameters():
            # do not prune biases and frozen layers
            if (not parameter_name.endswith("weight")) \
                    or (frozen_layers and parameter_name in frozen_layers):
                continue
            # clamp the weights
            parameter_group.data = torch.clamp(parameter_group.data, min=-range_clip, max=range_clip)

def get_model_size(model: nn.Module) -> float:
    tmp_filepath = join(".", "tmp.pth")
    torch.save(model.state_dict(), tmp_filepath)
    size_MB = getsize(tmp_filepath) / 1e6
    remove(tmp_filepath)
    return size_MB
