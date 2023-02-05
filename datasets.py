from os.path import isdir, join

import torch
import torchvision

from models import vgg, resnet


def parse_model(dataset_name: str, model_name: str, weights_path: str, return_frozen_layers: bool = False):
    assert isdir(weights_path), f"unable to find {weights_path}"
    assert isinstance(return_frozen_layers, bool)
    if dataset_name == "cifar10":
        if model_name == "resnet18":
            model = resnet.resnet18(pretrained=False)
            model.load_state_dict(torch.load(join(weights_path, "cifar10", "resnet18.pt")))
            frozen_layers = "fc"
        elif model_name == "resnet34":
            model = resnet.resnet34(pretrained=False)
            model.load_state_dict(torch.load(join(weights_path, "cifar10", "resnet34.pt")))
            frozen_layers = "fc"
        elif model_name == "resnet50":
            model = resnet.resnet50(pretrained=False)
            model.load_state_dict(torch.load(join(weights_path, "cifar10", "resnet50.pt")))
            frozen_layers = "fc"
        elif model_name == "vgg":
            model = vgg.vgg16_bn(pretrained=False)
            model.load_state_dict(torch.load(join(weights_path, "cifar10", "vgg16_bn.pt")))
            frozen_layers = "classifier.6"
        else:
            raise NotImplementedError(f"unrecognized model {model_name} for dataset {dataset_name}")
    elif dataset_name == "imagenet":
        if model_name == "resnet18":
            model = torchvision.models.resnet18()
            model.load_state_dict(torch.load(join(weights_path, "imagenet", "resnet18.pth")))
            frozen_layers = "fc"
        elif model_name == "resnet34":
            model = torchvision.models.resnet34()
            model.load_state_dict(torch.load(join(weights_path, "imagenet", "resnet34.pth")))
            frozen_layers = "fc"
        elif model_name == "resnet50":
            model = torchvision.models.resnet50()
            model.load_state_dict(torch.load(join(weights_path, "imagenet", "resnet50.pth")))
            frozen_layers = "fc"
        elif model_name == "mobilenet":
            model = torchvision.models.mobilenet_v3_large()
            model.load_state_dict(torch.load(join(weights_path, "imagenet", "mobilenet_v3_large.pth")))
            frozen_layers = "classifier.3"
        elif model_name == "vit":
            model = torchvision.models.vit_b_32()
            model.load_state_dict(torch.load(join(weights_path, "imagenet", "vit_b_32.pth")))
            frozen_layers = "heads.head"
        elif model_name == "swin":
            model = torchvision.models.swin_b()
            model.load_state_dict(torch.load(join(weights_path, "imagenet", "swin_b.pth")))
            frozen_layers = "head"
        else:
            raise NotImplementedError(f"unrecognized model {model_name} for dataset {dataset_name}")
    elif dataset_name == "coco":
        if model_name == "faster_rcnn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
            model.load_state_dict(torch.load(join(weights_path, "coco", "fasterrcnn_resnet50_fpn.pth")))
            frozen_layers = "roi_heads.box_predictor"
        else:
            raise NotImplementedError(f"unrecognized model {model_name} for dataset {dataset_name}")
    else:
        raise NotImplementedError(f"there are no models for dataset {dataset_name}")
    if return_frozen_layers:
        return model, frozen_layers
    else:
        return model
