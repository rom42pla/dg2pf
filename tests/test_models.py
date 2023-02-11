from os.path import join

import torch
import torchvision

from datasets import parse_model
from models import resnet


def classification_model_asserts(model, input_shape, num_classes):
    sample = torch.randn([1, *input_shape])
    output = model(sample)
    assert output.shape == (1, num_classes)

# IMAGENET

def test_resnet18_cifar10():
    model = resnet.resnet18(pretrained=False)
    model.load_state_dict(torch.load(join("models", "../models/weights", "cifar10", "resnet18.pt")))
    classification_model_asserts(
        model=model,
        input_shape=[3, 32, 32],
        num_classes=10,
    )

def test_resnet34_cifar10():
    model = resnet.resnet34(pretrained=False)
    model.load_state_dict(torch.load(join("models", "../models/weights", "cifar10", "resnet34.pt")))
    classification_model_asserts(
        model=model,
        input_shape=[3, 32, 32],
        num_classes=10,
    )

def test_resnet50_cifar10():
    model = resnet.resnet50(pretrained=False)
    model.load_state_dict(torch.load(join("models", "../models/weights", "cifar10", "resnet50.pt")))
    classification_model_asserts(
        model=model,
        input_shape=[3, 32, 32],
        num_classes=10,
    )

# IMAGENET

def test_resnet18_imagenet():
    model, frozen_layers = parse_model(dataset_name="imagenet", model_name="resnet18",
                                       weights_path=join("models", "../models/weights"), return_frozen_layers=True)
    classification_model_asserts(
        model=model,
        input_shape=[3, 224, 224],
        num_classes=1000,
    )

def test_vit_b_16_imagenet():
    model, frozen_layers = parse_model(dataset_name="imagenet", model_name="vit_b_16",
                                       weights_path=join("models", "../models/weights"), return_frozen_layers=True)
    classification_model_asserts(
        model=model,
        input_shape=[3, 384, 384],
        num_classes=1000,
    )

def test_vit_b_32_imagenet():
    model, frozen_layers = parse_model(dataset_name="imagenet", model_name="vit_b_32",
                                       weights_path=join("models", "../models/weights"), return_frozen_layers=True)
    classification_model_asserts(
        model=model,
        input_shape=[3, 224, 224],
        num_classes=1000,
    )

def test_swin_s_imagenet():
    model, frozen_layers = parse_model(dataset_name="imagenet", model_name="swin_s",
                                       weights_path=join("models", "../models/weights"), return_frozen_layers=True)
    classification_model_asserts(
        model=model,
        input_shape=[3, 224, 224],
        num_classes=1000,
    )

def test_deit_s_imagenet():
    model, frozen_layers = parse_model(dataset_name="imagenet", model_name="deit_s",
                                       weights_path=join("models", "../models/weights"), return_frozen_layers=True)
    classification_model_asserts(
        model=model,
        input_shape=[3, 224, 224],
        num_classes=1000,
    )

def test_deit_b_imagenet():
    model, frozen_layers = parse_model(dataset_name="imagenet", model_name="deit_b",
                                       weights_path=join("models", "../models/weights"), return_frozen_layers=True)
    classification_model_asserts(
        model=model,
        input_shape=[3, 224, 224],
        num_classes=1000,
    )
