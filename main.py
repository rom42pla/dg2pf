import argparse
import logging
import random
from copy import deepcopy
from typing import Iterable

import time
from datetime import datetime
from os import makedirs
from os.path import join, isdir
from pprint import pprint

import torch
import torchvision
from torchvision.transforms import transforms as T

from datasets import parse_model
from models import resnet, vgg
from disprunq import disprunq
from utils import save_json, get_model_size, set_seed

if __name__ == "__main__":
    torch.jit.enable_onednn_fusion(True)
    logging.getLogger().setLevel(logging.INFO)
    available_models = {"vgg", "mobilenet", "resnet18", "resnet34", "resnet50", "vit", "swin", "faster_rcnn"}

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="the name of the dataset",
                        type=str, choices={"imagenet", "cifar10", "coco"}, required=True)
    parser.add_argument("--dataset_path", help="path to the dataset",
                        type=str, required=True)
    parser.add_argument("--model_name", help="the name of the model to use",
                        type=str, choices=available_models)
    parser.add_argument("--teacher_name", help="the name of the teacher to use",
                        type=str, choices=available_models | {"same", None}, default=None)
    parser.add_argument("--logs_path", help="path to the logs",
                        type=str, default=join(".", "logs"))
    parser.add_argument("--weights_path", help="path to the weights",
                        type=str, default=join(".", "models", "weights"))
    parser.add_argument("--label", help="an optional label for the experiment folder",
                        type=str, default=None)

    parser.add_argument("--device", help="the device to use for training",
                        type=str, choices={"cpu", "gpu", "cuda", "auto"}, default="auto")
    parser.add_argument("--batch_size", help="the batch size for the dataloaders",
                        type=int, default=128)
    parser.add_argument("--pruning_percent", help="the percentage of weights to be pruned",
                        type=float, default=0.95)
    parser.add_argument("--layerwise_pruning", help="whether to toggle layerwise pruning",
                        action="store_true", default=False)
    parser.add_argument("--quantization_bits", help="the number of bits to be used for quantization",
                        type=int, default=32)
    parser.add_argument("--r", help="the value of r in the quantization function",
                        type=float, default=0.001)

    parser.add_argument("--learning_rate", help="the learning rate of the optimizer",
                        type=float, default=1e-4)
    parser.add_argument("--seed", help="the seed of the run for reproducibility purposes",
                        type=int, default=random.randint(0, int(9e6)))

    args = vars(parser.parse_args())
    assert isdir(args['dataset_path']), f"directory {args['dataset_path']} does not exists"
    if args['device'] == "gpu":
        args['device'] = "cuda"
    elif args['device'] == "auto":
        args['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    if args['device'] == "cuda" and not torch.cuda.is_available():
        raise Exception(f"no GPUs detected")
    assert args['batch_size'] >= 1
    assert 0 <= args['pruning_percent'] < 1
    assert args['quantization_bits'] >= 1
    if not isdir(args['logs_path']):
        makedirs(args['logs_path'])
    assert args['learning_rate'] > 0

    set_seed(args['seed'])

    if args['dataset_name'] == "imagenet":
        task = "classification"
        dataset_train = torchvision.datasets.ImageNet(args['dataset_path'],
                                                      split="train")
        dataset_val = torchvision.datasets.ImageNet(args['dataset_path'],
                                                    split="val")
        transforms_train = T.Compose([
            T.ToTensor(),
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
        transforms_val = T.Compose([
            T.ToTensor(),
            T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(224),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
    elif args['dataset_name'] == "cifar10":
        task = "classification"
        dataset_train = torchvision.datasets.CIFAR10(args['dataset_path'],
                                                     train=True, download=False)
        dataset_val = torchvision.datasets.CIFAR10(args['dataset_path'],
                                                   train=False, download=False)
        transforms_train = T.Compose([
            T.ToTensor(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
        transforms_val = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)),
        ])
    elif args['dataset_name'] == "coco":
        task = "object_detection"
        dataset_train = torchvision.datasets.CocoDetection(root=join(args['dataset_path'], "train2017"),
                                                           annFile=join(args['dataset_path'], "annotations",
                                                                        "instances_train2017.json"))
        dataset_val = torchvision.datasets.CocoDetection(root=join(args['dataset_path'], "val2017"),
                                                         annFile=join(args['dataset_path'], "annotations",
                                                                      "instances_val2017.json"))
        transforms_train = transforms_val = T.Compose([
            T.ToTensor(),
        ])
    else:
        raise NotImplementedError(f"unrecognized dataset {args['dataset']}")

    # todo fix the logger
    try:
        logFormatter = logging.Formatter(fmt='%(asctime)s\t%(message)s',
                                         datefmt='%Y-%m-%d %H:%M', )
        logging.getLogger().handlers[0].setFormatter(logFormatter)
    except Exception as e:
        logging.warning(f"there were some {type(e)} exception while setting up the logger")
        logging.warning(e)

    # sets up the logging folder
    experiment_path = join(args['logs_path'], f"{args['dataset_name']}_{datetime.today().strftime('%Y%m%d_%H%M')}")
    if args['label']:
        experiment_path = f"{experiment_path}_{args['label']}"
    if experiment_path is not None and not isdir(experiment_path):
        makedirs(experiment_path)
        logging.info(f"created folder {experiment_path}")
    pprint(args)
    save_json(args, join(experiment_path, "console_args.json"))
    time.sleep(0.5)

    # retrieves the models
    model, frozen_layers = parse_model(dataset_name=args['dataset_name'], model_name=args['model_name'],
                                       weights_path=args['weights_path'], return_frozen_layers=True)
    if args['teacher_name']:
        if task not in {"classification"}:
            logging.warning(f"distillation is implemented only for classification datasets, "
                            f"and {args['dataset_name']} is not")
            teacher = None
        if args['teacher_name'] == "same":
            args['teacher_name'] = args['model_name']
        teacher = parse_model(dataset_name=args['dataset_name'], model_name=args['teacher_name'],
                              weights_path=args['weights_path'])
    else:
        teacher = None

    if isinstance(frozen_layers, str):
        frozen_layers = {frozen_layers}
    frozen_layers_s = set()
    for frozen_layer in (frozen_layers if isinstance(frozen_layers, Iterable) else [frozen_layers]):
        frozen_layers_s = frozen_layers_s.union({parameter_name
                                                 for parameter_name, _ in model.named_parameters()
                                                 if parameter_name.startswith(frozen_layer)})
    frozen_layers = list(frozen_layers_s)

    # transfer the model(s) to the selected device
    model = model.to(args['device'])
    if teacher:
        teacher = teacher.to(args['device'])

    # launch disprunq
    disprunq(
        model=model,
        teacher=teacher,
        quantization_bits=args['quantization_bits'],
        pruning_percent=args['pruning_percent'],
        layerwise_pruning=args['layerwise_pruning'],
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        transforms_train=transforms_train,
        transforms_val=transforms_val,
        task=task,
        batch_size=args['batch_size'],
        frozen_layers=list(frozen_layers),
        logs_path=experiment_path,
        r=args['r'],
    )
