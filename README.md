# FedCycleBGD

The implementation for [Save It All: Enabling Full Parameter Tuning for Federated Large Language Models via Cycle Block Gradient Descent](https://arxiv.org/abs/2406.11187).
## Overview

This repository introduces our latest work **FedCycleBGD** which aims to explore the use of full-parameter tuning Language Models (LLMs) in the federated learning setting, with the objective of reducing communication, computation, and memory costs through the implementation of a cycle block update mechanism.
## Table of Contents

- [Introduction](#introduction)
- [Code Content](#code-content)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Acknowledgements](#acknowledgements)

## Introduction

The advent of large language models (LLMs) has revolutionized the deep learning paradigm, yielding impressive results across a wide array of tasks. However, the pre-training or fine-tuning of LLMs within a federated learning (FL) framework poses substantial challenges, including considerable computational and memory resource demands, as well as communication bottlenecks between servers and clients. Existing solutions either make the unrealistic assumption that the entire model is exchanged for training, or apply parameter-effective fine-tuning methods from centralized learning to train LLMs in FL which tend to underperform during training or fine-tuning stages due to the limited search subspace of parameter updating. In this paper, we introduce a novel method for the efficient training and fine-tuning of LLMs in FL, with minimal resource consumption. Our approach, termed FedCyBGD, utilizes Cycle Block Gradient Descent to periodically update the model. In particular, we design a compression scheme for FedCyBGD, aiming to further decrease the model download cost. It enables full parameter training in FL with only selected block updates and uploads, thereby reducing communication, computation, and memory costs. Our method achieves state-of-the-art performance for FL LLM training, while significantly reducing associated costs. Codes are provided here.

### Key Contributions

- **Save It All**: Cycle Block Update framework for full parameter tuning in FL.
- **Scalability**: Compression for model download and block upload.
- **Experiments**: Multiple empirical evaluation with resource consumption reduction.

## Code Content 

- Directory Structure
  - causal_llms
  - gpt
  - roberta-glue

- Usage
  - Evaluating General LLM SFT Generation Tasks
  - Evaluating SQuAD QA Tasks
  - Evaluating GLUE Benchmark Classification Tasks

## SetUp

Clone this repository to your local machine and install the required dependencies.

```bash
git clone https://github.com/L3030/FedCyBGD.git
cd FedCyBGD
pip install -r requirements.txt
```

## Quick Start
### Causal LLMs SFT Task
Here is a simple example for running experiments based on Llama-2-7b.
```bash
cd causal_llms
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage fl_sft \
    --model_name_or_path Llama-2-7b \
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
    --eval_steps 50 \
    --evaluation_strategy steps \
    --learning_rate 1e-6 \
    --num_train_epochs 3 \
    --overwrite_output_dir \
    --plot_loss \
    --switch_mode ascending \
    --bcd_epochs 6 \
    --num_clients 64 \
```

### SQuAD Task Using GPT2-Small
Here is a simple example for running experiments based on GPT2.
```bash
cd gpt
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset_name squad \
    --max_seq_length 384 \
    --model_name_or_path gpt2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --pad_to_max_length \
    --learning_rate 3e-5 \
    --num_clients 24 \
    --ft_type random
```


### GLUE Benchmark Using Roberta-base
Here is a simple example for running experiments based on Roberta-base.
```bash
cd roberta-glue
UDA_VISIBLE_DEVICES=0 python roberta.py \
--task_name mrpc \
--model_name_or_path roberta-base \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--block_type fl \
--num_clients 24
```

### Citation
```bash
@article{wang2024save,
  title={Save It All: Enabling Full Parameter Tuning for Federated Large Language Models via Cycle Black Gradient Descent},
  author={Wang, Lin and Wang, Zhichao and Tang, Xiaoying},
  journal={arXiv preprint arXiv:2406.11187},
  year={2024}
}
```


# Acknowledgements

FedCyBGD's BlockOptimizer solution mainly references [BAdam](https://github.com/Ledzy/BAdam), for which we are grateful. And we mainly follow [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 
to conduct our causal_llms experiments. We would like to express our gratitude to the contributors.


