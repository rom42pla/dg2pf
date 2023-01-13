import numpy as np
from datasets import load_dataset
import torchvision.transforms as T


def transforms_train(examples):
    transforms = T.Compose([
        T.RandomCrop(32),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    examples["pixel_values"] = [
        transforms(image.convert("RGB"))
        for image in examples["img"]
    ]
    del examples["img"]
    return examples


def transforms_val(examples):
    transforms = T.Compose([
        T.CenterCrop(32),
        T.ToTensor(),
    ])

    examples["pixel_values"] = [
        transforms(image.convert("RGB"))
        for image in examples["img"]
    ]
    del examples["img"]
    return examples


def load():
    # loads the dataset
    dataset_train = load_dataset("cifar10", split="train", ignore_verifications=True)
    dataset_val = load_dataset("cifar10", split="test", ignore_verifications=True)

    # applies the transforms
    dataset_train.set_transform(transforms_train)
    dataset_val.set_transform(transforms_val)
    return dataset_train, dataset_val
