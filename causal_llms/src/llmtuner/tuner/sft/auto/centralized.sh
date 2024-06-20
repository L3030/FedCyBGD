conda activate wzcllm
cd ../home/wzc/BAdam-master/llama-alpaca

nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type block \
    --output_dir ./outputs/llama2-7b/alpaca_gpt4_en/BAdam \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 9000 \
    --val_size 500 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 100 \
    --switch_mode random\
    --bcd_epochs 6 \
     > rc.log 2>&1' &

nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage sft \
    --model_name_or_path meta-llama/Llama-2-7b \
    --do_train \
    --dataset oaast_sft \
    --template default \
    --finetuning_type block \
    --output_dir ./outputs/llama2-7b/oaast_sft/BAdam \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_steps 9000 \
    --val_size 500 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_block_every 100 \
    --switch_mode ascending\
    --bcd_epochs 6 \
     > oc.log 2>&1' &

