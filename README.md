# Continual-Learning-Benchmark ResCL

This is fork of https://github.com/GT-RIPL/Continual-Learning-Benchmark with implemented ResCL

To run ResCL benchmark on CIFAR100 use run iBatchLearn.py with this command line:
--dataset CIFAR100 --train_aug --gpuid 0 --repeat 1 --incremental_class --optimizer Adam  --force_out_dim 100 --no_class_remap --first_split_size 20 --other_split_size 20 --schedule 20 30 32 --batch_size 128 --model_name WideResNet_28_2_cifar --model_type resnet --agent_type customization --agent_name ResCL --lr 0.001 --reg_coef 2