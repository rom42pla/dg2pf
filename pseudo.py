from typing import Union

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
                    for parameter_group in model.parameters()
                ])
            )
        ).detach().cpu().item()
    )
    return max_abs_weight


def prune(
        model: nn.Module,
        R: Union[int, float],
        beta: Union[int, float] = 50,
        percent: float = 0.925,
) -> nn.Module:
    assert beta > 0
    assert 0 < percent < 1
    # https://www.desmos.com/calculator/qmev0zpopf
    fn = lambda x: torch.exp(-(x * beta) ** 2)

    # todo set weights to 0
    print("ok")
    # for parameter_group in model.parameters():
    #     for parameter in parameter_group.flatten():
    #         if parameter == 0:
    #             parameter.require_grads = False
    pass


def train(
        model: nn.Module,
        dataset,
        tau: Union[int, float],
        alpha: Union[int, float] = 0.5,
) -> nn.Module:
    assert 0 <= alpha <= 1

    # todo implement training loop
    logits_t, logits_s, y = None, None, None
    loss = (
                   alpha * nn.KLDivLoss()(logits_t, logits_s) + (1 - alpha) * nn.CrossEntropyLoss()(logits_s, y)
           ) * (tau ** 2)


def train_quantize(
        model: nn.Module,
        dataset,
        tau: Union[int, float],
        b: int,
        R: Union[int, float],
        alpha: Union[int, float] = 0.5,
        r: float = 0.001,
) -> nn.Module:
    assert isinstance(b, int)
    assert b >= 1
    assert 0 < r < 1
    assert R > 0
    # todo implement quantization
    # todo implement training loop
    logits_t, logits_s, y = None, None, None
    loss = (
                   alpha * nn.KLDivLoss()(logits_t, logits_s) + (1 - alpha) * nn.CrossEntropyLoss()(logits_s, y)
           ) * (tau ** 2)


def disprunq(
        model: nn.Module,
        dataset,
        b: int,
        r: float = 0.001,
):
    R: float = get_max_abs_weight(model=model)

    pruned_model: nn.Module = prune(model, R=R)
    finetuned_pruned_model: nn.Module = train(
        model,
        dataset
    )
    finetuned_pruned_quantized_model: nn.Module = train_quantize(
        model,
        dataset,
        b=b,
        R=R,
        r=r,
    )


from transformers import ViTFeatureExtractor, ViTForImageClassification
import torchvision.transforms as T
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = T.Compose([
    T.ToTensor()
])(Image.open(requests.get(url, stream=True).raw)).unsqueeze(0)


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


disprunq(
    model=TestModel(),
    dataset=None,
    b=8,
    r=0.001,
)
