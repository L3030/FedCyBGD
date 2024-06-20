conda activate wzcllm
cd ../mnt/nvme1n1/wzc/BAdam-master/causal_llms

nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage fl_sft \
    --model_name_or_path bloom-1b7 \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type flblock \
    --output_dir ./outputs/llama2-7b \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 9000 \
    --val_size 500 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 300 \
    --switch_mode fixed \
    --bcd_epochs 6 \
    --num_clients 64 \
    --compress 0.5 \
     > alpaca_fixed.log 2>&1' &


nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage fl_sft \
    --model_name_or_path meta-llama/Llama-2-7b \
    --do_train \
    --dataset oaast_sft \
    --template default \
    --finetuning_type flblock \
    --output_dir ./outputs/llama2-7b \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 9000 \
    --val_size 500 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 300 \
    --switch_mode fixed \
    --bcd_epochs 6 \
    --num_clients 64 \
     > oaast_fixed.log 2>&1' &


nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage fl_sft \
    --model_name_or_path meta-llama/Llama-2-7b \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --output_dir ./outputs/llama2-7b \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 9000 \
    --val_size 500 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --learning_rate 1e-3 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 300 \
    --switch_mode fixed \
    --bcd_epochs 6 \
    --num_clients 64 \
    --compress 0.7 \
    --lora_target "q_proj, v_proj, k_proj, o_proj, up_proj, down_proj" \
     > 0.7.log 2>&1' &
"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\