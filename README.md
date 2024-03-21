Hey there!

Just a quick note to let you know that the code is actually out there. But, give us about a week or two (from today, 12/03/2024) because we're planning to spruce it up a bit and put together a readme for you. We'll kick off the repo and circle back with you in two weeks!

Update 1 (21/03/2024):
# Distilled Gradual Pruning with Pruned Fine-tuning

<p align="center">
<a href="https://ieeexplore.ieee.org/document/10438214" alt="arXiv">
</p>

---

This is a PyTorch implementation of the **Distilled Gradual Pruning with Pruned Fine-tuning (DG2PF)** algorithm proposed in our paper "[Distilled Gradual Pruning with Pruned Fine-tuning (DG2PF)](10.1109/TAI.2024.3366497)".

**Note**: DG2PF focuses on the optimization of neural networks through knowledge distillation and magnitude-based pruning, improving efficiency without compromising the performance of pre-trained networks.

<p align="center">
  <img src="algo.jpg" alt="DG2PF"/>
</p>

Figure 1: **Overview of DG2PF** showing the algorithm proposed for magnitude-based pruning and knowledge distillation to optimize pre-trained neural networks effectively.


## Image Classification
### 1. Requirements

- PyTorch (version >= 1.7.0)
- torchvision (version >= 0.8.0)
- pyyaml

For detailed requirements, refer to `requirements.txt` which can be installed using the following command:

```bash
pip install -r requirements.txt
```

Data preparation: <insert instructions or script for dataset setup>

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
python main.py --train --dataset_name <dataset> --model_name <model_name> --batch_size <size> --epochs <num_epochs>
```

### 5. Validate

Validation command example:

```bash
python main.py --validate --checkpoint <path_to_checkpoint>
```

## Acknowledgment
We would like to acknowledge the contributions of <acknowledgements>. This work is supported by <supporting parties>.

```
