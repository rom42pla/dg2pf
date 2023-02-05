import os
import logging
import re
import time
from copy import deepcopy
from os import makedirs
from os.path import join, isdir
from pprint import pprint
from typing import Union, Optional, List, Dict, Iterable, Any

import numpy as np
from PIL.Image import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from tqdm import tqdm

import torch

from pruning import get_pruned_perc, prune
from quantization import weight_quantize, quantization_fn, get_quantized_weight_model, Multiply, \
    check_if_quantization_works
from utils import save_json, coco_collate_fn, get_model_device, weight_clamp, get_classification_metrics, \
    get_object_detection_metrics, get_max_abs_weight, get_flattened_weights, get_model_size


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        task: str,
        quantization_bits: int,
        range_clip: float,
        do_quantization: bool,

        optimizer,

        teacher: Optional[nn.Module] = None,
        distillation_temperature: Union[int, float] = 4,
        distillation_alpha: Union[int, float] = 0.75,
        frozen_layers: Optional[Union[str, List[str]]] = None,

        input_scaler: Optional[torch.Tensor] = None,
        input_shifter: Optional[torch.Tensor] = None,

        r: float = 0.001,
        pruning_percent: float = 0.9,

        label: str = "",
):
    assert isinstance(quantization_bits, int)
    assert 0 < quantization_bits <= 32
    assert range_clip > 0
    assert 0 < r <= 1
    assert isinstance(do_quantization, bool)
    assert (distillation_temperature := float(distillation_temperature)) > 0
    assert 0 <= (distillation_alpha := float(distillation_alpha)) <= 1
    device = get_model_device(model)

    # updates the parameters' state
    model.train()
    if teacher is not None:
        teacher.eval()

    scaler = torch.cuda.amp.GradScaler()
    losses: List[torch.Tensor] = []
    for i_batch, batch in enumerate(progress_bar := tqdm(dataloader, desc="", leave=True)):
        if task == "classification":
            inputs, labels = batch[0].to(device), batch[1].to(device)
            assert len(inputs) == len(labels)
        elif task == "object_detection":
            inputs = [image.to(device)
                      for image, _ in batch]
            targets = [
                {k: v.to(device) for k, v in targets.items()}
                for _, targets in batch
            ]
            assert len(inputs) == len(targets)

        optimizer.zero_grad(set_to_none=True)

        if input_shifter:
            inputs = inputs + input_scaler
        if input_scaler:
            inputs = inputs * input_scaler

        with torch.autocast(device_type=device, dtype=torch.float16):
            # eventually quantize the model
            if do_quantization:
                with torch.no_grad():
                    weight_quantize(
                        model=model,
                        r=r,
                        bits=quantization_bits,
                        range_clip=range_clip,
                        frozen_layers=frozen_layers,
                    )
                    weight_clamp(model=model, range_clip=range_clip, frozen_layers=frozen_layers)

            if task == "classification":
                preds = model(inputs)
                loss = F.cross_entropy(preds, labels)
                if teacher is not None:
                    preds_log_probs = F.log_softmax(preds / distillation_temperature, dim=-1)
                    preds_teacher_log_probs = F.log_softmax(teacher(inputs) / distillation_temperature, dim=-1)

                    kl_div = F.kl_div(preds_log_probs, preds_teacher_log_probs,
                                      log_target=True, reduction="batchmean")
                    loss = (distillation_alpha * kl_div +
                            (1 - distillation_alpha) * loss) * \
                           distillation_temperature ** 2
            elif task == "object_detection":
                preds = model(inputs, targets)
                loss = sum([v for v in preds.values()])
                inputs = torch.stack([
                    T.Compose([
                        T.Resize(224),
                        T.CenterCrop(224),
                    ])(image)
                    for image in inputs
                ])
            loss_input = F.mse_loss(inputs, quantization_fn(inputs, 1, quantization_bits, range_clip)) * 1000 + \
                         F.mse_loss(torch.max(torch.abs(inputs)), torch.tensor(range_clip).to(device))
            loss = loss + loss_input

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        for parameter_name, parameter_group in model.named_parameters():
            if (not parameter_name.endswith("weight")) \
                    or (frozen_layers and parameter_name in frozen_layers):
                continue
            zero_mask = torch.abs(parameter_group.data) <= 1e-7
            parameter_group.grad[zero_mask] *= 0
            parameter_group.data[zero_mask] = 0

        scaler.step(optimizer)
        scaler.update()

        # computes the metrics
        # batch_metrics = get_classification_metrics(y_pred=labels_pred.float(),
        #                                            y_gt=labels.long())
        # batch_metrics['mean_loss'] = loss.detach().cpu()
        # for k in batch_metrics.keys():
        #     value = batch_metrics[k].detach().cpu().item()
        #     if k not in metrics.keys():
        #         metrics[k] = [value]
        #     else:
        #         metrics[k] += [value]
        losses += [loss.detach().cpu()]

        # updates the progress bar
        progress_bar.set_description(f"TRAIN {label} - "
                                     f"loss {losses[-1]:.3f}, "
                                     f"r {r:.3f}, "
                                     # f"acc@1 {np.mean(metrics['acc@1'][-16:]):.1f}%, "
                                     f"pruned_perc {pruning_percent * 100:.1f}%, "
                                     f"quantization_bits {quantization_bits}")
    return np.mean(losses)


def test_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        task: str,
        label: str = ""):
    assert task in {"classification", "object_detection"}
    assert isinstance(dataloader, DataLoader)
    device = get_model_device(model)

    model.eval()
    main_metric_names = {
        "classification": "acc@1",
        "object_detection": "map",
    }
    y_pred_list, y_gt_list = [], []
    for i_batch, batch in enumerate(progress_bar := tqdm(dataloader, desc="", leave=True)):
        # sets up the inputs
        if task == "classification":
            inputs, labels = batch[0].to(device), batch[1].cpu().tolist()
            y_gt_list += labels
        elif task == "object_detection":
            inputs = [image.to(device) for image, _ in batch]
            targets = [
                {k: v.cpu() for k, v in targets.items()}
                for _, targets in batch
            ]
            assert len(inputs) == len(targets)
            y_gt_list += targets

        # computes the inference
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
            preds = model(inputs)

        # saves the outputs
        if task == "classification":
            y_pred_list += preds.cpu().tolist()
        elif task == "object_detection":
            y_pred_list += [
                {k: v.cpu() for k, v in sample_preds.items()}
                for sample_preds in preds
            ]

        # computes the metrics
        if i_batch % 100 == 0 or i_batch == len(dataloader) - 1:
            if task == "classification":
                # metrics = get_classification_metrics(y_pred=torch.cat(y_pred_list, dim=0).float(),
                #                                      y_gt=torch.cat(y_gt_list, dim=0).long())
                metrics = get_classification_metrics(
                    y_pred=torch.as_tensor(y_pred_list, device="cpu", dtype=torch.float32),
                    y_gt=torch.as_tensor(y_gt_list, device="cpu", dtype=torch.long))
            elif task == "object_detection":
                metrics = get_object_detection_metrics(y_pred_list, y_gt_list)
            metrics['pruned_perc'] = get_pruned_perc(model) * 100

        # updates the progress bar
        progress_bar.set_description(f"TEST {label} - "
                                     f"{main_metric_names[task]} {metrics[main_metric_names[task]]:.1f}%, "
                                     f"pruned_perc {metrics['pruned_perc']:.1f}%")
    return {
        k: v.item() if isinstance(v, torch.Tensor) else v
        for k, v in metrics.items()
    }


def disprunq(
        model: nn.Module,
        dataset_train,
        dataset_val,
        task: str,

        transforms_train=None,
        transforms_val=None,
        r: float = 0.001,
        quantization_bits: int = 32,
        pruning_percent: float = 0.,
        layerwise_pruning: bool = False,
        teacher: Optional[nn.Module] = None,
        frozen_layers: Optional[Union[str, List[str]]] = None,

        learning_rate: float = 1e-4,
        batch_size: int = 1,
        num_workers: int = max(1, os.cpu_count() - 1),

        max_epochs: int = 100,
        early_stop_patience: int = 2,
        logs_path: Optional[str] = None,
):
    assert task in {"classification", "object_detection"}
    if task != "classification" and teacher is not None:
        raise NotImplementedError(f"distillation is implemented only for classification tasks")

    assert isinstance(quantization_bits, int) and quantization_bits >= 1
    assert 0 <= pruning_percent < 1
    if frozen_layers is None:
        frozen_layers = []
    if isinstance(frozen_layers, str):
        frozen_layers = [frozen_layers]
    for frozen_layer in frozen_layers:
        assert len({
            parameter_name
            for parameter_name, _ in model.named_parameters()
            if re.fullmatch(f"{frozen_layer}.*", parameter_name)
        }) >= 1, f"layer {frozen_layer} not found into model"
    assert learning_rate > 0
    assert isinstance(batch_size, int) and batch_size >= 1
    assert isinstance(num_workers, int) and num_workers >= 0

    assert isinstance(max_epochs, int) and max_epochs >= 1
    assert isinstance(early_stop_patience, int) and early_stop_patience >= 1

    assert logs_path is None or isinstance(logs_path, str)
    if logs_path is not None and not isdir(logs_path):
        makedirs(logs_path)
        logging.info(f"created folder {logs_path}")
    for dataset in [dataset_train, dataset_val]:
        if task == "classification":
            assert isinstance(dataset[0], tuple) and len(dataset[0]) == 2 \
                   and isinstance(dataset[0][0], Image) and isinstance(dataset[0][1], int), \
                f"the datasets must contain tuples of PIL images and int labels"
        elif task == "object_detection":
            # todo add asserts for object detection
            pass

    # sets up the model and the quantized model
    device: str = get_model_device(model)
    if device != "cpu":
        model = model.cpu()
    quantized_model = deepcopy(model)
    model, quantized_model = model.to(device), quantized_model.to(device)

    range_clip: float = get_max_abs_weight(model=model, frozen_layers=frozen_layers)
    input_scaler = nn.Parameter(torch.ones(1, device=device, requires_grad=True))
    input_shifter = nn.Parameter(torch.zeros(1, device=device, requires_grad=True))

    # sets up the optimizer
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': input_shifter},
        {'params': input_scaler},
    ], lr=1e-5, weight_decay=0)

    # sets up the datasets
    main_metric_names = {
        "classification": "acc@1",
        "object_detection": "map",
    }
    if task == "classification":
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True)
        if transforms_train:
            dataset_train.transform = transforms_train
        if transforms_val:
            dataset_val.transform = transforms_val
    elif task == "object_detection":
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size,
                                      shuffle=True, num_workers=1,
                                      pin_memory=True, collate_fn=lambda batch: list(map(coco_collate_fn,
                                                                                         [sample
                                                                                          for sample in batch
                                                                                          if len(sample[1]) >= 1])
                                                                                     )
                                      )
    logs_summary: Dict[str, Union[List[str], Dict[str, Any]]] = {
        "quantization_bits": quantization_bits,
        "metrics_base_model": test_one_epoch(
            model=model,
            dataloader=DataLoader(dataset_val,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  collate_fn=None if task == "classification" else lambda batch: list(
                                      map(coco_collate_fn,
                                          [sample
                                           for sample in batch
                                           if len(sample[1]) >= 1])
                                  )),
            task=task,
            label="base".upper(),
        ),
        "base_model_parameters": get_flattened_weights(model).numel(),
        "base_model_size_MB": get_model_size(model),
    }
    if len(frozen_layers) >= 1:
        logs_summary['frozen_layers'] = frozen_layers
    if teacher is not None:
        logs_summary['metrics_teacher'] = test_one_epoch(
                model=teacher,
                task=task,
                dataloader=DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=True),
                label="teacher".upper(),
            )
        logs_summary['teacher_parameters'] = get_flattened_weights(model).numel()
    last_score, epochs_without_improvement = 0, 0
    logs_training: List[Dict[str, Dict[str, float]]] = []

    if pruning_percent > 0:
        prune(
            model=model,
            pruning_percent=pruning_percent,
            frozen_layers=frozen_layers,
            layerwise=layerwise_pruning,
        )
    logs_summary['pruned_model_parameters'] = get_flattened_weights(model, frozen_layers=frozen_layers).count_nonzero().item()
    logs_summary['pruning_percent'] = 1 - logs_summary['pruned_model_parameters'] / logs_summary['base_model_parameters']
    pprint(logs_summary)

    for epoch in range(max_epochs):
        time.sleep(0.1)

        # recompute the range clip after the first quantization epoch
        if epoch == 2:
            range_clip = get_max_abs_weight(model=model, frozen_layers=frozen_layers)
            last_score, epochs_without_improvement = 0, 0

        train_one_epoch(
            model=model,
            teacher=teacher,
            dataloader=dataloader_train,
            task=task,
            do_quantization=True if epoch >= 1 else False,
            optimizer=optimizer,
            quantization_bits=quantization_bits,
            pruning_percent=pruning_percent,
            range_clip=range_clip,
            label=f"epoch {epoch}".upper(),
            frozen_layers=frozen_layers,
            input_scaler=input_scaler,
            input_shifter=input_shifter,
            r=r,
        )

        # todo fix vgg giving wrong quantization
        quantized_model = get_quantized_weight_model(model=model, quantized_model=quantized_model,
                                                     quantization_bits=quantization_bits, range_clip=range_clip,
                                                     frozen_layers=frozen_layers)
        if not 'quantized_model_size_MB' in logs_summary:
            logs_summary['quantized_model_size_MB'] = get_model_size(quantized_model)
        pprint(logs_summary)
        exit()

        dataset_val.transform = T.Compose([
            transforms_val,
            Multiply(input_scaler.item(), input_shifter.item(),
                     quantization_bits, range_clip,
                     do_quantization=True)
        ])
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

        metrics_val_epoch_unquantized = test_one_epoch(
            model=model,
            task=task,
            dataloader=dataloader_val,
            label=f"epoch {epoch} - model not quantized".upper(),
        )

        metrics_val_epoch_quantized = test_one_epoch(
            model=quantized_model,
            task=task,
            dataloader=dataloader_val,
            label=f"epoch {epoch} - model quantized".upper(),
        )
        assert metrics_val_epoch_unquantized.keys() == metrics_val_epoch_quantized.keys()

        # saves the logs
        logs_training += [{
            "metrics_val_epoch_unquantized": metrics_val_epoch_unquantized,
            "metrics_val_epoch_quantized": metrics_val_epoch_quantized,
        }]
        if logs_path is not None:
            save_json(logs_training, path=join(logs_path, "disprunq_logs_training.json"))

        # updates early stopping
        if last_score <= metrics_val_epoch_quantized[main_metric_names[task]]:
            last_score = metrics_val_epoch_quantized[main_metric_names[task]]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # checks if early stopping criteria is met
        if epochs_without_improvement >= early_stop_patience and teacher is not None:
            logging.info("teacher detached".upper())
            teacher = None
            last_score = metrics_val_epoch_quantized[main_metric_names[task]]
            epochs_without_improvement = 0
        elif epochs_without_improvement >= early_stop_patience:
            logging.info(f"early stop of {early_stop_patience} epochs met".upper())
            break
        print()
    logs_summary.update({
        "quantized_model": {
            k: np.max([logs_epoch['metrics_val_epoch_quantized'][k]
                       for logs_epoch in logs_training])
            for k, v in logs_summary["metrics_base_model"].items()
        },
        "unquantized_model": {
            k: np.max([logs_epoch['metrics_val_epoch_unquantized'][k]
                       for logs_epoch in logs_training])
            for k, v in logs_summary["metrics_base_model"].items()
        },
    })
    pprint(logs_summary)
    if logs_path is not None:
        save_json(logs_summary, path=join(logs_path, "disprunq_logs_summary.json"))
