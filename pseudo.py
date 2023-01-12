import os
import pickle
from typing import Union, Optional

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.functional as F


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
    margin = 1e-5
    while get_pruned_perc(model)[0] < old_perc + percent:
        margin += 0.0001
        for parameter_name, parameter_group in model.named_parameters():
            # prune weights only, not biases
            if 'weight' in parameter_name:
                # prunes weights near to the zero of the gaussian
                parameter_group.data[torch.abs(parameter_group.data) < margin] = 0


# def train(
#         model: nn.Module,
#         dataset,
#         tau: Union[int, float],
#         alpha: Union[int, float] = 0.5,
# ) -> nn.Module:
#     assert 0 <= alpha <= 1
#
#     # todo implement training loop
#     logits_t, logits_s, y = None, None, None
#     loss = (
#                    alpha * nn.KLDivLoss()(logits_t, logits_s) + (1 - alpha) * nn.CrossEntropyLoss()(logits_s, y)
#            ) * (tau ** 2)

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

    for batch_idx, (inputs, targets) in enumerate(dataloader_train):
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
        print(batch_idx, len(dataloader_train),
              'Loss: %.3f | Acc: %.3f (%d/%d)  | perc Pruned: %.3f'
              % (
                  train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                  get_pruned_perc(model)[0]))


def test_one_epoch(
        model: nn.Module,
        dataloader_test,
        current_epoch: int,
        criterion,
        device: str = "auto"):
    assert current_epoch >= 0
    assert device in {"cpu", "cuda", "auto"}
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader_test):
            inputs, targets = inputs.to(device), \
                              targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # todo fix progress bar
            print(batch_idx, len(dataloader_test), 'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                  % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': current_epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


# def train_quantize(
#         model: nn.Module,
#         dataset,
#         tau: Union[int, float],
#         b: int,
#         R: Union[int, float],
#         alpha: Union[int, float] = 0.5,
#         r: float = 0.001,
# ) -> nn.Module:
#     assert isinstance(b, int)
#     assert b >= 1
#     assert 0 < r < 1
#     assert R > 0
#     # todo implement quantization
#     # todo implement training loop
#     logits_t, logits_s, y = None, None, None
#     loss = (
#                    alpha * nn.KLDivLoss()(logits_t, logits_s) + (1 - alpha) * nn.CrossEntropyLoss()(logits_s, y)
#            ) * (tau ** 2)


def disprunq(
        model: nn.Module,
        dataset,
        quantization_bits: int,
        percent: float,
        r: float = 0.001,
        max_epochs: int = 100,
        teacher: Optional[nn.Module] = None,
):
    assert isinstance(max_epochs, int)
    assert max_epochs >= 1
    range_clip = get_max_abs_weight(model=model)
    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0)
    for epoch in range(max_epochs):
        if epoch == 1:
            range_clip = get_max_abs_weight(model=model)
        train_one_epoch(
            model=model,
            teacher=teacher,
            dataloader_train=None,
            do_pruning=True if epoch == 0 else False,
            do_quantization=False if epoch == 0 else True,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            quantization_bits=quantization_bits,
            percent=percent,
            range_clip=range_clip,
            current_epoch=epoch,
        )
        print("test float:")
        test_one_epoch(
            model=model,
            dataloader_test=None,
            current_epoch=epoch,
            criterion=nn.CrossEntropyLoss(),
        )
        print("test quantized:")

        quantized_model = pickle.loads(
            pickle.dumps(model)
        )
        with torch.no_grad():
            quantized_model.apply(WeightClipper(range_clip))
            weight_quantize(quantized_model, 1, bits=quantization_bits, range_clip=range_clip)
            quantized_model.apply(WeightClipper(range_clip))

        test_one_epoch(
            model=quantized_model,
            dataloader_test=None,
            current_epoch=epoch,
            criterion=nn.CrossEntropyLoss(),
        )
        print("Unique values in weights: Float:", get_unique_weights(model).shape[0],
              "Quantized(<2^", quantization_bits, "):", get_unique_weights(quantized_model).shape[0])
        check_if_quantization_works(quantized_model, range_clip, quantization_bits)

    # pruned_quantized_model: nn.Module = train_quantize(
    #     model,
    #     dataset,
    #     b=b,
    #     R=R,
    #     r=r,
    # )
    # return pruned_quantized_model


from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision.transforms as T
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = T.Compose([
    T.ToTensor()
])(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0)


class WeightClipper(object):

    def __init__(self, range_clip):
        self.range_clip = range_clip

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.range_clip, +self.range_clip)
            module.weight.data = w


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.classifier = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def forward(self):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.classifier(**inputs)
        logits = outputs.logits
        return logits


if __name__ == "__main__":
    model = TestModel()
    disprunq(
        model=model,
        teacher=TestModel(),
        dataset=None,
        quantization_bits=8,
        r=0.001,
        percent=92.5,
    )
