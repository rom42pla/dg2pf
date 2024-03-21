Hey there!

Just a quick note to let you know that the code is actually out there. But, give us about a week or two (from today, 12/03/2024) because we're planning to spruce it up a bit and put together a readme for you. We'll kick off the repo and circle back with you in two weeks!

Update 1 (21/03/2024):
Hey there!

Just a quick note to let you know that the code is actually out there. But, give us about a week or two (from today, 12/03/2024) because we're planning to spruce it up a bit and put together a readme for you. We'll kick off the repo and circle back with you in two weeks!

Update 1 (21/03/2024):
# Distilled Gradual Pruning with Pruned Fine-tuning

<p align="center">
<a href="https://ieeexplore.ieee.org/document/10438214" alt="arXiv">
</p>

---

This is a PyTorch implementation of the **Distilled Gradual Pruning with Pruned Fine-tuning (DG2PF)** algorithm proposed in our paper "[Distilled Gradual Pruning with Pruned Fine-tuning (DG2PF)](https://ieeexplore.ieee.org/document/10438214)".

**Note**: DG2PF focuses on the optimization of neural networks through knowledge distillation and magnitude-based pruning, improving efficiency without compromising the performance of pre-trained networks.

<p align="center">
  <img src="algo.jpg" alt="DG2PF"/>
</p>

Figure 1: **Overview of DG2PF** showing the algorithm proposed for magnitude-based pruning and knowledge distillation to optimize pre-trained neural networks effectively.


## Image Classification
### 1. Requirements

For detailed requirements, refer to `requirements.txt` which can be installed using the following command:

```bash
pip install -r requirements.txt
```

Data preparation: <insert instructions or script for dataset setup>
```markdown
### Data Preparation

For ImageNet data preparation, please organize the dataset into the following directory structure:

```
│ImageNet/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── ...
│   ├── ...
├── val/
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   ├── ILSVRC2012_val_00002138.JPEG
│   │   ├── ...
│   ├── ...
```

You can obtain the ImageNet dataset from the official [ImageNet website](http://www.image-net.org/) or other sources that provide the dataset. Ensure you follow the rules and regulations of the dataset provider.

If you require a script to assist in downloading or arranging the ImageNet dataset, you may use the following script:
https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4

### 2. Models

<p align="center">
  <img src="res.jpg" alt="DG2PF"/>
</p>

### 3. Usage

For usage instructions and examples, refer to `main.py`. Example command to run a model:

```bash
python main.py --dataset_name <dataset> --model_name <model_name> --batch_size <size>
```

Replace `<dataset>`, `<model_name>`, and `<size>` with appropriate values.

### 4. Train

Training command example:

```bash
python main.py --dataset_name <dataset> --model_name <model_name> --teacher_name <teacher_name> --pruning_percent <pruning perc>
```
--dataset_name "imagenet" or "cifar10"
--model_name  and --teacher_name {"vgg", "mobilenet", "resnet18", "resnet34", "resnet50", "vit", "vit_b_16", "vit_b_32", "swin_s", "deit_s", "deit_b", "faster_rcnn"}
--pruning_percent float > 0 and < 1

You can se other parameters top of main.py

to reproduce training with resnet 50 use:
```bash
python main.py --dataset_name "imagenet" --model_name "resnet50" --teacher_name "same" --pruning_percent 0.9
```


