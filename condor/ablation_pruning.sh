script_starter="bash condor_train.sh"
shared_params="--seed=42 --dataset_name=imagenet --dataset_path=/data1/visionlab/romeo/disprunq/datasets/imagenet2012 --weights_path=/data1/visionlab/romeo/disprunq/models/weights --logs_path=/data1/visionlab/romeo/disprunq/logs --model_name=vit --teacher_name=same --quantization_bits=8"

$script_starter $shared_params --pruning_percent=0.1 --label="ablation_pruning_0.1"
$script_starter $shared_params --pruning_percent=0.2 --label="ablation_pruning_0.2"
$script_starter $shared_params --pruning_percent=0.3 --label="ablation_pruning_0.3"
$script_starter $shared_params --pruning_percent=0.5 --label="ablation_pruning_0.5"
$script_starter $shared_params --pruning_percent=0.7 --label="ablation_pruning_0.7"
$script_starter $shared_params --pruning_percent=0.9 --label="ablation_pruning_0.9"
$script_starter $shared_params --pruning_percent=0.95 --label="ablation_pruning_0.95"

condor_q