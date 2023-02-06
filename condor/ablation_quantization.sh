script_starter="bash condor_train.sh"
shared_params="--seed=42 --dataset_name=imagenet --dataset_path=/data1/visionlab/romeo/disprunq/datasets/imagenet2012 --weights_path=/data1/visionlab/romeo/disprunq/models/weights --logs_path=/data1/visionlab/romeo/disprunq/logs --teacher_name=same --pruning_percent=0.5"

$script_starter $shared_params --model_name=vit --quantization_bits=1 --label="ablation_quantization_vit_1"
$script_starter $shared_params --model_name=vit --quantization_bits=2 --label="ablation_quantization_vit_2"
$script_starter $shared_params --model_name=vit --quantization_bits=3 --label="ablation_quantization_vit_3"
$script_starter $shared_params --model_name=vit --quantization_bits=4 --label="ablation_quantization_vit_4"
$script_starter $shared_params --model_name=vit --quantization_bits=8 --label="ablation_quantization_vit_8"

$script_starter $shared_params --model_name=resnet18 --quantization_bits=1 --label="ablation_quantization_resnet18_1"
$script_starter $shared_params --model_name=resnet18 --quantization_bits=2 --label="ablation_quantization_resnet18_2"
$script_starter $shared_params --model_name=resnet18 --quantization_bits=3 --label="ablation_quantization_resnet18_3"
$script_starter $shared_params --model_name=resnet18 --quantization_bits=4 --label="ablation_quantization_resnet18_4"
$script_starter $shared_params --model_name=resnet18 --quantization_bits=8 --label="ablation_quantization_resnet18_8"

condor_q