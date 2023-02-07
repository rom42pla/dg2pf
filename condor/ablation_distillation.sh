script_starter="bash condor_train.sh"
shared_params="--seed=42 --dataset_name=imagenet --dataset_path=/data1/visionlab/romeo/disprunq/datasets/imagenet2012 --weights_path=/data1/visionlab/romeo/disprunq/models/weights --logs_path=/data1/visionlab/romeo/disprunq/logs --pruning_percent=0.5 --quantization_bits=8"

$script_starter $shared_params --model_name=vit --teacher_name=vit --label="ablation_distillation_vit_vit"
$script_starter $shared_params --model_name=vit --teacher_name=resnet18 --label="ablation_distillation_vit_resnet18"
$script_starter $shared_params --model_name=vit --label="ablation_distillation_vit_none"

$script_starter $shared_params --model_name=resnet18 --teacher_name=resnet18 --label="ablation_distillation_resnet18_resnet18"
$script_starter $shared_params --model_name=resnet18 --teacher_name=vit --label="ablation_distillation_resnet18_vit"
$script_starter $shared_params --model_name=resnet18 --label="ablation_distillation_resnet18_none"

condor_q