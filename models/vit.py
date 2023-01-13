import os

from torch import nn
import torchvision.transforms as T
from transformers import ViTForImageClassification


class ViT(nn.Module):
    def __init__(
            self,
            dataset: str = "imagenet"
    ):
        super().__init__()
        assert dataset in {"imagenet", "cifar10"}
        # os.environ["CURL_CA_BUNDLE"] = ""
        if dataset == "imagenet":
            self.hf_model_name = "google/vit-base-patch16-224"
        else:
            self.hf_model_name = "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10"
        self.model = ViTForImageClassification.from_pretrained(self.hf_model_name)

    def forward(self, images):
        if images.shape[2:] != [224, 224]:
            images = T.Compose([
                T.Resize([224, 224]),
            ])(images)
        logits = self.model(images).logits
        return logits
