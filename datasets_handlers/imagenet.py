import huggingface_hub
import numpy as np
from datasets import load_dataset
import torchvision.transforms as T


def transforms_train(examples):
    transforms = T.Compose([
        T.RandomCrop(224),
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    examples["pixel_values"] = [
        transforms(image.convert("RGB"))
        for image in examples["img"]
    ]
    return examples


def transforms_val(examples):
    transforms = T.Compose([
        T.CenterCrop(224),
        T.Resize(224),
        T.ToTensor(),
    ])

    examples["pixel_values"] = [
        transforms(image.convert("RGB"))
        for image in examples["img"]
    ]
    return examples


def load():
    # logs into the hub
    huggingface_hub.login()

    # loads the dataset
    dataset_train = load_dataset("imagenet-1k", split="train", ignore_verifications=True)
    # dataset_val = load_dataset("imagenet-1k", split="validation", ignore_verifications=True)
    dataset_val = dataset_train
    # applies the transforms
    dataset_train.set_transform(transforms_train)
    dataset_val.set_transform(transforms_val)
    return dataset_train, dataset_val
