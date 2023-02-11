script_starter="bash condor_train.sh"
shared_params="--seed=42 --dataset_name=imagenet --dataset_path=/data1/visionlab/romeo/disprunq/datasets/imagenet2012 --weights_path=/data1/visionlab/romeo/disprunq/models/weights --logs_path=/data1/visionlab/romeo/disprunq/logs --teacher_name=same"

$script_starter $shared_params --model_name=deit_s --quantization_bits=32 --pruning_percent=0.4 --label="results_imagenet_deit_s_32q_0.4p"
$script_starter $shared_params --model_name=deit_s --quantization_bits=8 --pruning_percent=0.4 --label="results_imagenet_deit_s_8q_0.4p"

condor_q