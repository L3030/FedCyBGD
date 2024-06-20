conda activate wzcllm
cd ../mnt/nvme1n1/wzc/BAdam-master/gpt

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --pad_to_max_length \
    --learning_rate 3e-5 \
    --num_clients 24 \
    --ft_type random \
     > squad_fixed.log 2>&1' &

nohup bash -c 'CUDA_VISIBLE_DEVICES=6 python train.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --pad_to_max_length \
    --learning_rate 1e-3 \
    --num_clients 24 \
    --ft_type lora \
     > squad_lora.log 2>&1' &

nohup bash -c 'CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --pad_to_max_length \
    --learning_rate 3e-5 \
    --num_clients 24 \
    --ft_type badam \
     > squad_badam.log 2>&1' &

nohup bash -c 'CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --pad_to_max_length \
    --learning_rate 1e-3 \
    --num_clients 24 \
    --ft_type adapter \
     > squad_ada.log 2>&1' &


nohup bash -c 'CUDA_VISIBLE_DEVICES=5 python train.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --pad_to_max_length \
    --learning_rate 1e-3 \
    --num_clients 24 \
    --ft_type full \

     > squad_ada.log 2>&1' &