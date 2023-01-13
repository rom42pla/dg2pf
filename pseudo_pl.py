import gc
import os
import pickle
from copy import deepcopy
from typing import Union, Optional

import einops
import huggingface_hub
from datasets import load_dataset
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy
from torchvision.transforms import transforms
import pytorch_lightning as pl
from tqdm import tqdm, trange

from datasets_handlers import cifar10, imagenet
from distillation import KDLoss
from models.base_classification import BaseClassificationModel
from models.resnet import ResNet, resnet18
from models.vggbase import VGGBase, VGG
from models.vit import ViT
from pruning import prune, get_pruned_perc, get_max_abs_weight, get_unique_weights
from quantization import WeightClipper, weight_quantize, check_if_quantization_works


def train_one_epoch(
        model: nn.Module,
        dataloader_train,
        current_epoch: int,
        quantization_bits: int,
        range_clip: float,
        do_quantization: bool,
        do_pruning: bool,
        optimizer,
        criterion,
        teacher: Optional[nn.Module] = None,
        r: float = 0.001,
        beta: Union[int, float] = 50,
        percent: float = 0.925,
        device: str = "auto",
):
    assert isinstance(current_epoch, int)
    assert current_epoch >= 0
    assert isinstance(quantization_bits, int)
    assert 0 < quantization_bits <= 32
    assert range_clip > 0
    assert 0 < r < 1
    assert isinstance(do_quantization, bool)
    assert isinstance(do_pruning, bool)
    assert device in {"cpu", "cuda", "auto"}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print('Epoch: %d' % current_epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    clipper = WeightClipper(range_clip)

    if do_pruning:
        prune(
            model=model,
            beta=beta,
            percent=percent
        )

    pbar = tqdm(dataloader_train, desc="", leave=True)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), \
            targets.to(device)

        if do_quantization:
            with torch.no_grad():
                weight_quantize(
                    model=model,
                    r=r,
                    bits=quantization_bits,
                    range_clip=range_clip
                )
                # todo replace apply
                model.apply(clipper)

        # computes the loss for the student
        optimizer.zero_grad()
        outputs = model(inputs)
        if teacher is None:
            loss = criterion(outputs, targets)
        else:
            with torch.no_grad():
                outputs_distill = teacher(inputs)
            loss = KDLoss()(outputs, targets,
                            outputs_distill, 4, 0.75)  # torch.nn.MSELoss()(outputs,outputs_distill)
        # loss = loss_partial #+ loss_distill
        loss.backward()

        for parameter_name, parameter_group in model.named_parameters():
            if 'weight' in parameter_name:
                zero_mask = torch.abs(parameter_group.data) <= 1e-7
                parameter_group.grad[zero_mask] *= 0
                parameter_group.data[zero_mask] = 0

        optimizer.step()

        # computing accuracy
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(net)
        # todo adjust progress bar

        # print(batch_idx, len(dataloader_train),
        #     'Loss: %.3f | Acc: %.3f (%d/%d)  | perc Pruned: %.3f'
        #    % (
        #       train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
        #      get_pruned_perc(model)[0]))"
        pbar.set_description('TRAIN - Loss: %.3f | Acc: %.3f (%d/%d)  | perc Pruned: %.3f'
                             % (
                                 train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                                 get_pruned_perc(model)[0]))


def test_one_epoch(
        model: nn.Module,
        dataloader_test,
        current_epoch: int,
        criterion,
        is_quantized: bool = False,
        device: str = "auto"):
    assert current_epoch >= 0
    assert device in {"cpu", "cuda", "auto"}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    pbar = tqdm(dataloader_test, desc="", leave=True)

    losses = []
    labels_pred, labels_gt = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            inputs, targets = batch["pixel_values"].to(device), \
                batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            labels_pred += [outputs]
            labels_gt += [targets]
            losses += [loss]

            accuracy = multiclass_accuracy(torch.cat(labels_pred, dim=0),
                                           torch.cat(labels_gt, dim=0),
                                           num_classes=labels_pred[-1].shape[-1])
            f1 = multiclass_f1_score(torch.cat(labels_pred, dim=0),
                                     torch.cat(labels_gt, dim=0),
                                     num_classes=labels_pred[-1].shape[-1])
            mean_loss = torch.stack(losses).mean()

            label = "FLOAT    " if is_quantized is False else "QUANTIZED"
            pbar.set_description(f"TEST {label} - loss = {mean_loss:.3f} | accuracy: {accuracy:.3f} | f1: {f1:.3f}")


def disprunq(
        model: nn.Module,
        dataset_train,
        dataset_val,
        quantization_bits: int,
        pruning_percent: float,
        r: float = 0.001,
        do_pruning: bool = True,
        do_quantization: bool = True,
        max_epochs: int = 100,
        teacher: Optional[nn.Module] = None,
        batch_size: int = 2,
        device: str = "auto",
):
    assert isinstance(max_epochs, int)
    assert max_epochs >= 1
    assert isinstance(batch_size, int)
    assert batch_size >= 1
    assert device in {"cpu", "cuda", "auto"}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BaseClassificationModel(
        model=model,
        teacher=teacher,
        do_pruning=do_pruning,
        pruning_percent=pruning_percent,
        do_quantization=do_quantization,
        quantization_bits=quantization_bits,
    )
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() - 2)
    dataloader_test = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() - 2)

    trainer = pl.Trainer(
        accelerator="gpu" if device == "cuda" else "cpu",
        # devices=1 if device == "cuda" else os.cpu_count(),
        # check_val_every_n_epoch=1,
        # gradient_clip_val=1,
        # log_every_n_steps=1,
        precision=16,
        # callbacks=[],
        enable_model_summary=True,
    )
    trainer.test(model, dataloader_test)
    trainer.fit(model, dataloader_train, dataloader_test)
    exit()
    # test_one_epoch(
    #     model=model,
    #     dataloader_test=dataloader_test,
    #     current_epoch=0,
    #     criterion=nn.CrossEntropyLoss(),
    #     is_quantized=False,
    #     device=device,
    # )

    # for epoch in range(max_epochs):
    #     if epoch == 1:
    #         range_clip = get_max_abs_weight(model=model)
    #     train_one_epoch(
    #         model=model,
    #         teacher=teacher,
    #         dataloader_train=dataloader_train,
    #         do_pruning=True if (epoch == 0 and do_pruning) else False,
    #         do_quantization=False if epoch == 0 else True,
    #         optimizer=optimizer,
    #         criterion=nn.CrossEntropyLoss(),
    #         quantization_bits=quantization_bits,
    #         percent=pruning_percent,
    #         range_clip=range_clip,
    #         current_epoch=epoch,
    #         device=device,
    #     )
    #     # print("test float:")
    #     test_one_epoch(
    #         model=model,
    #         dataloader_test=dataloader_test,
    #         current_epoch=epoch,
    #         criterion=nn.CrossEntropyLoss(),
    #         is_quantized=False,
    #         device=device,
    #     )
    #     # print("test quantized:")
    #
    #     quantized_model = pickle.loads(
    #         pickle.dumps(model)
    #     )
    #     with torch.no_grad():
    #         quantized_model.apply(WeightClipper(range_clip))
    #         weight_quantize(quantized_model, 1, bits=quantization_bits, range_clip=range_clip)
    #         quantized_model.apply(WeightClipper(range_clip))
    #
    #     test_one_epoch(
    #         model=quantized_model,
    #         dataloader_test=dataloader_test,
    #         current_epoch=epoch,
    #         criterion=nn.CrossEntropyLoss(),
    #         is_quantized=True,
    #     )
    #     print("Unique values in weights: Float:", get_unique_weights(model).shape[0],
    #           "Quantized(<2^", quantization_bits, "):", get_unique_weights(quantized_model).shape[0])
    #     check_if_quantization_works(quantized_model, range_clip, quantization_bits)


if __name__ == "__main__":
    dataset = "cifar10"
    if dataset == "imagenet":
        dataset_train, dataset_val = imagenet.load()
    elif dataset == "cifar10":
        dataset_train, dataset_val = cifar10.load()
    else:
        raise Exception(f"unrecognized dataset {dataset}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "vit"
    assert model in {"resnet", "vit", "vgg"}
    if model == "resnet":
        # model = ResNet(dataset=dataset).to(device)
        # model = resnet18(pretrained=True).to(device)
        pass
    elif model == "vit":
        model = ViT(dataset=dataset).to(device)
    elif model == "vgg":
        model = VGG(dataset=dataset).to(device)
    disprunq(
        model=model,
        teacher=deepcopy(model).to(device),
        # teacher=VGG(dataset=dataset).to(device),
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        quantization_bits=8,
        r=0.001,
        pruning_percent=0.9,
        batch_size=128,
        do_pruning=True,
        do_quantization=False,
        device=device,
    )
