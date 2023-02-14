script_starter="bash condor_train.sh"
shared_params="--seed=42 --dataset_name=imagenet --dataset_path=/data1/visionlab/romeo/disprunq/datasets/imagenet2012 --weights_path=/data1/visionlab/romeo/disprunq/models/weights --logs_path=/data1/visionlab/romeo/disprunq/logs --teacher_name=same"

#$script_starter $shared_params --model_name=deit_s --batch_size=128 --quantization_bits=32 --pruning_percent=0.4 --label="results_imagenet_deit_s_32q_40p"
#$script_starter $shared_params --model_name=deit_s --batch_size=128 --quantization_bits=8 --pruning_percent=0 --label="results_imagenet_deit_s_8q_0p"
#$script_starter $shared_params --model_name=deit_s --batch_size=128 --quantization_bits=4 --pruning_percent=0 --label="results_imagenet_deit_s_4q_0p"
#$script_starter $shared_params --model_name=deit_s --batch_size=128 --quantization_bits=3 --pruning_percent=0 --label="results_imagenet_deit_s_3q_0p"
#$script_starter $shared_params --model_name=deit_s --batch_size=192 --quantization_bits=8 --pruning_percent=0.65 --label="results_imagenet_deit_s_8q_65p"

#$script_starter $shared_params --model_name=swin_s --batch_size=64 --quantization_bits=32 --pruning_percent=0.4 --label="results_imagenet_swin_s_32q_40p"
#$script_starter $shared_params --model_name=swin_s --batch_size=64 --quantization_bits=8 --pruning_percent=0 --label="results_imagenet_swin_s_8q_0p"
$script_starter $shared_params --model_name=swin_s --batch_size=64 --quantization_bits=8 --pruning_percent=0.65 --label="results_imagenet_swin_s_8q_65p"

#$script_starter $shared_params --model_name=deit_b --batch_size=96 --quantization_bits=32 --pruning_percent=0.4 --label="results_imagenet_deit_b_32q_40p"
#$script_starter $shared_params --model_name=deit_b --batch_size=96 --quantization_bits=8 --pruning_percent=0 --label="results_imagenet_deit_b_8q_0p"

#$script_starter $shared_params --model_name=resnet18 --batch_size=256 --quantization_bits=32 --pruning_percent=0.9 --label="results_imagenet_resnet18_32q_90p"
#$script_starter $shared_params --model_name=resnet18 --batch_size=256 --quantization_bits=8 --pruning_percent=0.9 --label="results_imagenet_resnet18_8q_90p"
$script_starter $shared_params --model_name=resnet18 --batch_size=256 --quantization_bits=8 --pruning_percent=0.65 --label="results_imagenet_resnet18_8q_65p"

condor_q