import gc
import math
import os
from typing import Optional, Dict, List, Any, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_accuracy
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision.transforms as T


class BaseClassificationModel(pl.LightningModule):
    def __init__(
            self,

            model: nn.Module,
            teacher: Optional[nn.Module] = None,

            do_pruning: bool = False,
            pruning_percent: float = 0.1,

            do_quantization: bool = False,
            quantization_bits: int = 8,
            r: float = 0.001,
    ):
        super().__init__()
        assert isinstance(do_pruning, bool)

        assert isinstance(model, nn.Module)
        self.model: nn.Module = model

        assert teacher is None or isinstance(teacher, nn.Module)
        self.teacher: Optional[nn.Module] = teacher
        if self.teacher:
            self.teacher.eval()
            for parameter in self.teacher.parameters():
                parameter.requires_grad = False

        assert isinstance(do_pruning, bool)
        self.do_pruning: bool = do_pruning
        if self.do_pruning:
            assert 0 < pruning_percent < 1
            self.pruning_percent: float = pruning_percent
        self.range_clip: float = self.get_max_abs_weight()

        assert isinstance(do_quantization, bool)
        self.do_quantization: bool = do_quantization
        if self.do_quantization:
            assert isinstance(quantization_bits, int) and quantization_bits >= 1
            self.quantization_bits: int = quantization_bits
            assert 0 < r < 1
            self.r: float = float(r)

        self.phase: Optional[str] = None
        self.queue: Dict[str, Dict[str, List[Any]]] = self.init_queue()

    def forward(self, images):
        outs = self.model(images)
        if self.teacher is not None and self.phase == "train":
            with torch.no_grad():
                outs_teacher = self.teacher(images)
            return outs, outs_teacher
        return outs

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        # optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        return optimizer

    def on_after_backward(self):
        for parameter_name, parameter_group in self.model.named_parameters():
            if 'weight' in parameter_name:
                mask = torch.isclose(parameter_group.data,
                                     torch.zeros_like(parameter_group.data),
                                     atol=1e-9)
                parameter_group.grad[mask] *= 0
                parameter_group.data[mask] = 0

    def step(self, batch):
        if self.do_quantization:
            with torch.no_grad():
                self.weight_quantize()
                # todo replace apply
                self.model.apply(WeightClipper(self.range_clip))

        inputs, label_gt = batch["pixel_values"].to(self.device), \
            batch["label"].to(self.device)
        if self.teacher is None or self.phase != "train":
            label_pred = self(inputs)
            loss = F.cross_entropy(label_pred, label_gt)
        else:
            label_pred, label_pred_teacher = self(inputs)
            loss = KDLoss()(label_pred, label_gt,
                            label_pred_teacher, 4, 0.75)
        del inputs
        return {
            "loss": loss,
            "label_gt": label_gt,
            "label_pred": label_pred,
        }

    def step_end(self, step_output):
        self.queue[self.phase]["loss"] += [step_output["loss"].detach().cpu()]
        self.queue[self.phase]["label_gt"] += [label for label in step_output["label_gt"].detach().cpu()]
        self.queue[self.phase]["label_pred"] += [label for label in step_output["label_pred"].detach().cpu()]

        mean_loss = torch.mean(torch.stack(self.queue[self.phase]["loss"]))
        self.log(f"mean_loss_{self.phase}", mean_loss, prog_bar=True)

        f1 = multiclass_f1_score(
            torch.stack(self.queue[self.phase]["label_pred"]),
            torch.stack(self.queue[self.phase]["label_gt"]),
            num_classes=len(self.queue[self.phase]["label_pred"][0]),
        )
        self.log(f"f1_{self.phase}", f1, prog_bar=True)

        accuracy = multiclass_accuracy(
            torch.stack(self.queue[self.phase]["label_pred"]),
            torch.stack(self.queue[self.phase]["label_gt"]),
            num_classes=len(self.queue[self.phase]["label_pred"][0]),
        )
        self.log(f"acc_{self.phase}", accuracy, prog_bar=True)

        # self.free_memory()

    def training_step(self, batch, batch_idx):
        outs = self.step(batch)
        return outs

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.step(batch)
        return outs

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            outs = self.step(batch)
        return outs

    def training_step_end(self, step_output):
        self.step_end(step_output)

    def validation_step_end(self, step_output):
        self.step_end(step_output)

    def test_step_end(self, step_output):
        self.step_end(step_output)

    def on_train_start(self) -> None:
        self.queue = self.init_queue()
        if self.current_epoch == 0 and self.do_pruning:
            self.prune()
        if self.current_epoch == 1:
            self.range_clip = self.get_max_abs_weight()

    def on_train_epoch_start(self) -> None:
        self.phase = "train"
        self.model.train()

    def on_validation_epoch_start(self) -> None:
        self.phase = "val"
        self.model.eval()

    def on_test_epoch_start(self) -> None:
        self.phase = "test"
        self.model.eval()

    def init_queue(self) -> Dict[str, Dict[str, List[Any]]]:
        return {
            phase: {
                "loss": [],
                "label_pred": [],
                "label_gt": []
            }
            for phase in ["train", "val", "test"]
        }

    def free_memory(self):
        gc.collect()

    def get_max_abs_weight(
            self,
    ) -> float:
        max_abs_weight: float = float(
            torch.max(
                torch.abs(
                    torch.cat([
                        parameter_group.flatten()
                        for parameter_name, parameter_group in self.model.named_parameters()
                        if "weight" in parameter_name
                    ])
                )
            ).detach().cpu().item()
        )
        return max_abs_weight

    def prune(self):
        # retrieves the weights of the model as a flattened tensor
        weights = self.get_flattened_weights()

        # finds the left and right margin for the pruning
        left_margin = np.percentile(weights, ((1 - self.pruning_percent) / 2) * 100)
        right_margin = np.percentile(weights, (self.pruning_percent + (1 - self.pruning_percent) / 2) * 100)
        assert right_margin >= left_margin
        del weights

        # prunes the weights
        for parameter_name, parameter_group in self.model.named_parameters():
            # do not prune the biases
            if 'weight' in parameter_name:
                mask = (parameter_group.data < right_margin) & \
                       (parameter_group.data > left_margin)
                parameter_group.data[mask] = 0
                del mask

        # checks that everything is alright
        pruned_weights = self.get_flattened_weights()
        assert ((pruned_weights < right_margin) &
                (pruned_weights > left_margin) &
                (pruned_weights != 0)) \
                   .nonzero().numel() == 0
        assert math.isclose(self.pruning_percent, self.get_pruned_perc(), abs_tol=0.001)
        del pruned_weights, left_margin, right_margin
        self.free_memory()

    def get_flattened_weights(self) -> torch.Tensor:
        weights = None
        for parameter_name, parameter_group in self.model.named_parameters():
            if 'weight' in parameter_name:
                if weights is None:
                    weights = parameter_group.data.clone().flatten().detach().cpu()
                else:
                    weights = torch.concatenate([weights, parameter_group.data.clone().flatten().detach().cpu()])
        return weights

    def get_pruned_perc(self):
        q = 0
        tot = 0
        for parameter_name, parameter_group in self.model.named_parameters():
            if 'weight' in parameter_name:
                mask = torch.abs(parameter_group.data) == 0

                q += torch.sum(mask).item()
                tot += torch.numel(parameter_group)
        return q / tot

    def weight_quantize(
            self,
    ):
        with torch.no_grad():
            for parameter_name, parameter_group in self.model.named_parameters():
                if "weight" in parameter_name:
                    # step = range_clip / torch.exp2(torch.tensor(bits - 1))
                    # mask = torch.abs(parameter_group.data) > 1e-7
                    mask = ~torch.isclose(parameter_group.data,
                                          torch.zeros_like(parameter_group.data),
                                          atol=1e-9)
                    # mask = torch.abs((W_.data // step) * step) > 0.00000001
                    parameter_group.data[~mask] = 0

                    if self.r == 1:
                        j_ = torch.exp2(torch.tensor(self.quantization_bits - 1)) / self.range_clip
                        parameter_group.data[mask] = (torch.ceil(parameter_group.data[mask] * j_) / j_) - \
                                                     (1 / (torch.exp2(
                                                         torch.tensor(self.quantization_bits)) / self.range_clip))
                    else:
                        j = torch.exp2(torch.tensor(self.quantization_bits)) / self.range_clip
                        pi = torch.pi

                        parameter_group.data[mask] = (
                                parameter_group.data[mask] - (
                                (torch.arctan(
                                    -(
                                            (self.r * torch.sin(j * pi * parameter_group.data[mask])) / (
                                            1 - (self.r * torch.cos(j * pi * parameter_group.data[mask])))))) / (
                                        pi * j * (1 / 2))))


class WeightClipper:

    def __init__(self, range_clip):
        self.range_clip = range_clip

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.range_clip, +self.range_clip)
            module.weight.data = w


class KDLoss(nn.Module):
    """Knowledge Distillation loss."""

    def __init__(self, dim: int = -1, scale_T: bool = True) -> None:
        """Initializer for KDLoss.

        Args:
            dim (int, optional): Dimension across which to apply loss. Defaults to -1.
            scale_T (bool, optional): Whether to scale kldiv by T^2. Defaults to True.
        """
        super().__init__()

        self.dim = dim
        self.scale_T = scale_T

    def forward(self, pred: torch.Tensor, target: torch.Tensor, teacher_pred: torch.Tensor, T: float, alpha: float,
                beta: float = None) -> torch.Tensor:
        """Forward method for KDLoss.

        Args:
            pred (torch.Tensor): Predictions of student model. Tensor of shape (batch, num_classes).
            target (torch.Tensor): Labels. LongTensor of shape (batch,), containing class integers like [1, 2, 3, ...].
            teacher_pred (torch.Tensor): Predictions of teacher model. Tensor of shape (batch, num_classes).
            T (float): Temperature value for evaluating softmax.
            alpha (float): Weight for kldiv.
            beta (float, optional): Weight for crossentropy. If not provided (beta=None), will use beta = 1 - alpha. Defaults to None.

        Returns:
            torch.Tensor: Loss value.
        """

        assert T >= 1.0, f"Expected temperature greater or equal to 1.0, but got {T}."

        if beta == None:
            assert alpha < 1.0, f"For weighted average (beta=None), alpha must be less than 1.0, but got {alpha}."
            beta = 1.0 - alpha

        # if self.scale_T:
        #    alpha = alpha

        pred_log_probs = F.log_softmax(pred / T, dim=self.dim)
        teacher_pred_log_probs = F.log_softmax(teacher_pred / T, dim=self.dim)

        kldiv = F.kl_div(pred_log_probs, teacher_pred_log_probs, log_target=True)
        crossentropy = F.cross_entropy(pred, target)

        return (alpha * kldiv + beta * crossentropy) * T * T
