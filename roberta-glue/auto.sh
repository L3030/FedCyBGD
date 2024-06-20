conda activate wzcllm
cd ../mnt/nvme1n1/wzc/galore

pkill wandb-service

CUDA_VISIBLE_DEVICES=0 python roberta.py \
--task_name mrpc \
--model_name_or_path roberta-base \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--block_type fl \
--num_clients 24 \
--full

CUDA_VISIBLE_DEVICES=7 python roberta.py \
--task_name mrpc \
--model_name_or_path roberta-base \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--block_type fl \
--num_clients 24

CUDA_VISIBLE_DEVICES=7 python roberta.py \
--task_name mrpc \
--model_name_or_path roberta-base \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--learning_rate 1e-4 \
--num_train_epochs 3 \
--block_type fl \
--num_clients 24 \
--lora


