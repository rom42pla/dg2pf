from os.path import join

import torch
import torchvision

from models import resnet

def classification_model_asserts(model, input_shape, num_classes):
    sample = torch.randn([1, *input_shape])
    output = model(sample)
    assert output.shape == (1, num_classes)

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

def test_resnet18_imagenet():
    model = torchvision.models.resnet18()
    model.load_state_dict(torch.load(join("models", "../models/weights", "imagenet", "resnet18.pth")))
    classification_model_asserts(
        model=model,
        input_shape=[3, 224, 224],
        num_classes=1000,
    )
