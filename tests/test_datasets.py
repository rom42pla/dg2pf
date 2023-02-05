from os.path import join

import torchvision
from PIL.Image import Image

def classification_sample_asserts(sample):
    assert isinstance(sample, tuple)
    assert isinstance(sample[0], Image)
    assert isinstance(sample[1], int)

def test_imagenet():
    for dataset in [
        torchvision.datasets.ImageNet(join("..", "..", "datasets", "vision", "imagenet2012"), split="train"),
        torchvision.datasets.ImageNet(join("..", "..", "datasets", "vision", "imagenet2012"), split="val"),
    ]:
        classification_sample_asserts(dataset[0])

def test_cifa10():
    for dataset in [
        torchvision.datasets.CIFAR10(join("..", "..", "datasets", "vision", "cifar10"), train=True, download=False),
        torchvision.datasets.CIFAR10(join("..", "..", "datasets", "vision", "cifar10"), train=False, download=False),
    ]:
        classification_sample_asserts(dataset[0])