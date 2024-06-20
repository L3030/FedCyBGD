# Inspired by: https://github.com/huggingface/transformers/blob/v4.29.2/examples/pytorch/summarization/run_summarization.py
import copy
import os
import random
import time


import pickle
import torch
import torch.distributed as dist
from collections import OrderedDict
from transformers.trainer import TRAINER_STATE_NAME
import json
import matplotlib.pyplot as plt
import numpy as np
from .utils import *
from typing import TYPE_CHECKING, Optional, List
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
import math
from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.misc import get_logits_processor
from llmtuner.extras.ploting import plot_loss
from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.tuner.sft.metric import ComputeMetrics
from llmtuner.tuner.sft.trainer import Seq2SeqPeftTrainer
from llmtuner.extras.logging import get_logger
import gc
import time
from .block_prefix import *
import os
os.environ["WANDB_DISABLED"] = "true"
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "custom_peft/src"))

import adapters
from adapters import AdapterArguments, AdapterTrainer, setup_adapter_training
from adapters import LoRAConfig

logger = get_logger(__name__)


if TYPE_CHECKING:
    from transformers import TrainerCallback
    from llmtuner.hparams import ModelArguments, DataArguments, FinetuningArguments, GeneratingArguments


def run_fl_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)

#----------------powersgd--------------------
    power_sgd = finetuning_args.power_sgd
    if power_sgd:
        training_args.optim = 'sgd'

    if finetuning_args.debug_mode:
        dataset = dataset.select(range(1000))
        print("Debug mode is enabled. Only 1000 samples are used for training.")
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
#---------------model_args------------------
    model_name = model_args.model_name_or_path.lower()
    if 'llama' in model_name:
        block_prefix = llama_block_prefix
        model_name = 'llama'
    elif 'chatglm' in model_name:
        block_prefix = chatglm_block_prefix
        model_name = 'chatglm'
    elif 'bloom' in model_name:
        block_prefix = bloom_block_prefix
        model_name = 'bloom'
    else:
        raise ValueError(f"Invalid model type to create proper block prefix")
# ----------------model dtype--------------------
    if 'adap' in finetuning_args.finetuning_type:
        if model_name == 'llama':
            adapters.init(model)
            if 'lora' in finetuning_args.finetuning_type:
                config = LoRAConfig(r=16, alpha=32)
                model.add_adapter("lora_adapter", config=config)
                model.train_adapter("lora_adapter")
            else:
                adapter_config = {'original_ln_before': True, 'original_ln_after': True, 'residual_before_ln': True,
                                  'adapter_residual_before_ln': False, 'ln_before': False, 'ln_after': False,
                                  'mh_adapter': False,
                                  'output_adapter': True, 'non_linearity': 'relu', 'reduction_factor': 32,
                                  'inv_adapter': None,
                                  'inv_adapter_reduction_factor': None, 'cross_adapter': False, 'leave_out': []}
                model.add_adapter("fedadapter", config=adapter_config)
                model.train_adapter("fedadapter")
        else:
            from custom_peft import (  # noqa: E402
                LoraConfig,
                BottleneckConfig,
                PrefixTuningConfig,
                get_peft_model,
                get_peft_model_state_dict,
                prepare_model_for_int8_training,
                set_peft_model_state_dict,
            )
            if 'lora' in finetuning_args.finetuning_type:
                config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=['query_key_value'],
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            else:
                config = BottleneckConfig(
                    bottleneck_size=128,
                    non_linearity='relu',
                    adapter_dropout=0.0,
                    use_parallel_adapter=False,
                    use_adapterp=False,
                    target_modules=['dense_4h_to_h'],
                    scaling=1.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                print('here ok')
            model = get_peft_model(model, config)
    if 'lora' in finetuning_args.finetuning_type:
        training_args.fp16 = True
    else:
        model.to(torch.bfloat16) if model_name == 'chatglm' else model.to(torch.float16)

# ----------------finetune args--------------------

    num_clients = finetuning_args.num_clients
    switch_mode = finetuning_args.switch_mode
    bcd_epochs = finetuning_args.bcd_epochs
    sample_num = finetuning_args.sample_num

    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.max_target_length,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    split_dataset_dic = split_dataset(dataset, data_args, training_args)
    total_train_dataset, total_eval_dataset = split_dataset_dic['train_dataset'], split_dataset_dic['eval_dataset']

# ----------------datasplit--------------------
    # total_train_dataset = total_train_dataset.select(range(0, 320)) # train_dataset for debug
    # total_eval_dataset = total_eval_dataset.select(range(0, 24))    # eval_dataset for debug
    par_data_num = len(total_train_dataset) // num_clients
    clients_data = []
    for i in range(num_clients):
        start_idx = i * par_data_num
        end_idx = start_idx + par_data_num if i < num_clients - 1 else len(total_train_dataset)
        subset = total_train_dataset.select(range(start_idx, end_idx))
        clients_data.append(subset)
    # ########## FL dataset split end###########

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = list(set([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids))
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

# ----------------dict-------------------
    block_results = OrderedDict()
    client2block = OrderedDict()
    block2client = OrderedDict()
    block_num = len(block_prefix)

    for layer_name in block_prefix:
        block_results[layer_name[0]] = []

    if switch_mode == 'ascending':
        for cid in range(num_clients):
            client2block[cid] = [cid % block_num for _ in range(bcd_epochs)]
    elif switch_mode == 'descending':
        for cid in range(num_clients):
            client2block[cid] = [(block_num-1) - (cid % block_num) for _ in range(bcd_epochs)]

    elif switch_mode == 'random':
        block_ids = [i for i in range(len(block_prefix))]
        for cid in range(num_clients):
            client2block[cid] = random.sample(block_ids, bcd_epochs)

    elif switch_mode == 'agg':
        assert sample_num is not None
        for bid in range(len(block_prefix)):
            block2client[bid] = [int(bid), int(block_num+bid)]

    # For saving config
    eval_results = []
# ----------------output path--------------------
    output_dic = f'./outputs/{model_name}/{data_args.dataset}/{num_clients}clients_{bcd_epochs}epochs_{switch_mode}_{training_args.optim}_' \
                 f'{finetuning_args.compress}_compress/'
    if switch_mode == 'agg':
        output_dic = output_dic.rstrip('/')
        output_dic += f'_{sample_num}sampled/'
    if finetuning_args.finetuning_type == 'lora':
        output_dic = output_dic.rstrip('/')
        output_dic += f'_lora/'
    if not os.path.exists(output_dic):
        os.makedirs(output_dic)
    output_path = output_dic + f'{switch_mode}_output.pkl'

# ----------------train--------------------
    if training_args.do_train:
        update_flag = False
        opt_flag = False
        current_state_dict = None
        log_stream = ""
        # with torch.autocast(device_type=training_args.device.type):
        for e in range(bcd_epochs):
            start_time = time.time()
            result_dict = {} # key~client_id, value~[block_id, result] for fixed and random
            if switch_mode in ['ascending', 'random', 'descending']:
                for i in range(num_clients):
                    client_model = copy.deepcopy(model)
                    if update_flag: # reload all updates if exists
                        update_global(client_model, block_results, part=finetuning_args.compress)
                    client_trainer = Seq2SeqPeftTrainer(
                        finetuning_args=finetuning_args,
                        client_bid=client2block[i][e],
                        client_id=i,
                        block_prefix_list=block_prefix,
                        model=client_model,
                        args=training_args,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        callbacks=callbacks,
                        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
                        # **split_dataset(clients_data[i], data_args, training_args)
                        train_dataset=clients_data[i],
                        eval_dataset=total_eval_dataset,
                    )

                    # client_trainer.args.output_dir = output_dic + f'/round{e}/client{i}/block{client_trainer.block_his}'
                    client_trainer.args.output_dir = output_dic
                    myckp = output_dic if opt_flag else None
                    train_result = client_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint, myckp=myckp, psgd=power_sgd)
                    # client_trainer.log_metrics("train", train_result.metrics)
                    # client_trainer.save_metrics("train", train_result.metrics)
                    if i == num_clients - 1:
                        client_trainer.save_state()
                        client_trainer.save_model()

                    # record the update
                    # block_results[block_prefix[client_trainer.block_his][0]] = get_block_param(client_model, client_trainer.block_his)
                    block_results[block_prefix[client2block[i][e]][0]] = get_block_param(client_model, client2block[i][e], block_prefix)
                    update_flag = True

                    if training_args.do_eval:
                        metrics = client_trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
                        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
                            metrics.pop("eval_loss", None)
                        # client_trainer.log_metrics("eval", metrics)
                        # client_trainer.save_metrics("eval", metrics)
                        result_dict[i] = [client2block[i][e], metrics['eval_loss']]
                        log_stream += outFunc(output_dir=output_dic, round=e, cid=i, bid=client2block[i][e], loss=metrics['eval_loss'], switch_mode=switch_mode)
                        save_log(output_dic, switch_mode, log=log_stream)

                    # if client_trainer.is_world_process_zero() and model_args.plot_loss:
                    #     plot_loss(client_trainer.args.output_dir, keys=["loss", "eval_loss"])

                    # save status of optimizer and lr_scheduler to stimulate cycle FL environment
                    client_trainer._save_optimizer_and_scheduler(output_dir=output_dic)
                    client_trainer._save_rng_state(output_dir=output_dic)
                    opt_flag = True

                    # free memory
                    client_trainer._freegpu()
                    del client_trainer
                    del client_model
                    del train_result
                    gc.collect()
                    torch.cuda.empty_cache()
                eval_results.append(result_dict)
                with open(output_path, 'wb') as f:
                    pickle.dump(eval_results, f)

            elif switch_mode in ['full']:
                for i in range(num_clients):
                    client_model = copy.deepcopy(model)
                    if update_flag: # reload all updates if exists
                        # Load the state dictionary into the model
                        client_model.load_state_dict(current_state_dict)

                    client_trainer = Seq2SeqPeftTrainer(
                        finetuning_args=finetuning_args,
                        client_id=i,
                        model=client_model,
                        args=training_args,
                        block_prefix_list=block_prefix,
                        tokenizer=tokenizer,
                        data_collator=data_collator,
                        callbacks=callbacks,
                        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
                        # **split_dataset(clients_data[i], data_args, training_args)
                        train_dataset=clients_data[i],
                        eval_dataset=total_eval_dataset,
                    )

                    client_trainer.args.output_dir = output_dic
                    myckp = output_dic if opt_flag else None
                    train_result = client_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint, myckp=myckp, psgd=power_sgd)
                    # client_trainer.log_metrics("train", train_result.metrics)
                    # client_trainer.save_metrics("train", train_result.metrics)

                    if (e+1) % 3 ==0 and i == num_clients - 1:
                        client_trainer.save_state()
                        client_trainer.save_model()

                    if training_args.do_eval:
                        metrics = client_trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
                        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
                            metrics.pop("eval_loss", None)
                        # client_trainer.log_metrics("eval", metrics)
                        # client_trainer.save_metrics("eval", metrics)
                        result_dict[i] = metrics['eval_loss']
                        log_stream += outFunc(output_dir=output_dic, round=e, cid=i, bid=None, loss=metrics['eval_loss'], switch_mode=switch_mode)
                    # if client_trainer.is_world_process_zero() and model_args.plot_loss:
                    #     plot_loss(client_trainer.args.output_dir, keys=["loss", "eval_loss"])

                    # save status of optimizer and lr_scheduler
                    # record the update
                    client_trainer._save_optimizer_and_scheduler(output_dir=output_dic)
                    client_trainer._save_rng_state(output_dir=output_dic)
                    current_state_dict = client_model.state_dict()
                    update_flag = True
                    opt_flag = True

                    # free memory
                    client_trainer._freegpu()
                    del client_trainer
                    del client_model
                    del train_result
                    gc.collect()
                    torch.cuda.empty_cache()
                eval_results.append(result_dict)
                with open(output_path, 'wb') as f:
                    pickle.dump(eval_results, f)

            elif switch_mode in ['agg']:
                for bidx, bid in enumerate(range(len(block_prefix))):
                    block_update = []
                    client_losses = []
                    for cidx, cid in enumerate(block2client[bid]):
                        client_model = copy.deepcopy(model)
                        if update_flag:
                            update_global(client_model, block_results)
                        client_trainer = Seq2SeqPeftTrainer(
                            finetuning_args=finetuning_args,
                            client_bid=bid,
                            client_id=cid,
                            model=client_model,
                            block_prefix_list=block_prefix,
                            args=training_args,
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            callbacks=callbacks,
                            compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
                            train_dataset=clients_data[cid],
                            eval_dataset=total_eval_dataset,
                        )
                        # client_trainer.args.output_dir = output_dic + f'round_{e}/block_{bid}_client_{cid}'
                        client_trainer.args.output_dir = output_dic
                        myckp = output_dic if opt_flag else None

                        if cidx == 0 and not (bidx == 0 and e == 0):
                            # 当前eval的是上一轮的结果，因此block往前推一个，如果当前block是0，推到上一个epoch的block 31
                            metrics = client_trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
                            if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
                                metrics.pop("eval_loss", None)
                            if bidx == 0: # 取上一轮的block 31的结果
                                c_losses = eval_results[e-1][len(block_prefix)-1]
                                record_round = e-1
                                record_bid = len(block_prefix)-1
                            else:
                                c_losses = result_dict[bidx-1]
                                record_round = e
                                record_bid = bid-1
                            c_losses.append(metrics['eval_loss'])
                            log_stream += outFunc(output_dic, round=record_round, cid=block2client[record_bid], bid=record_bid, loss=c_losses, switch_mode=switch_mode)
                            save_log(output_dic, switch_mode, log=log_stream)

                        # start training
                        train_result = client_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint, myckp=myckp, psgd=power_sgd)

                        # client_trainer.log_metrics("train", train_result.metrics)
                        # client_trainer.save_metrics("train", train_result.metrics)

                        # record the update
                        block_update.append(get_block_param(client_model, bid, block_prefix))

                        if training_args.do_eval:
                            metrics = client_trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
                            if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
                                metrics.pop("eval_loss", None)
                            # client_trainer.log_metrics("eval", metrics)
                            # client_trainer.save_metrics("eval", metrics)
                            client_losses.append(metrics['eval_loss'])


                        client_trainer._save_optimizer_and_scheduler(output_dir=output_dic)
                        client_trainer._save_rng_state(output_dir=output_dic)
                        opt_flag = True

                        # if client_trainer.is_world_process_zero() and model_args.plot_loss:
                        #     plot_loss(client_trainer.args.output_dir, keys=["loss", "eval_loss"])

                        if not (e == bcd_epochs - 1 and bidx == len(block_prefix) - 1 and cidx == len(sample_num) - 1):
                            # free memory
                            client_trainer._freegpu()
                            del client_trainer
                            del client_model
                            del train_result
                            gc.collect()
                            torch.cuda.empty_cache()

                    result_dict[bid] = client_losses
                    block_results[block_prefix[bid][0]] = aggregate_params(block_update)
                    update_flag = True
                    if e == bcd_epochs - 1 and bidx == len(block_prefix) - 1:
                        metrics = client_trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
                        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
                            metrics.pop("eval_loss", None)
                        # client_trainer.log_metrics("eval", metrics)
                        # client_trainer.save_metrics("eval", metrics)
                        c_losses = result_dict[bidx - 1]
                        c_losses.append(metrics['eval_loss'])
                        log_stream += outFunc(output_dic, round=e, cid=block2client[bid-1],
                                              bid=bid-1, loss=c_losses, switch_mode=switch_mode)
                        save_log(output_dic, switch_mode, log=log_stream)

                eval_results.append(result_dict)
                with open(output_path, 'wb') as f:
                    pickle.dump(eval_results, f)

            # if finetuning_args.finetuning_type in ['']
            end_time = time.time()
            log_stream += f"{num_clients}clients, round{e}, {switch_mode}, round_time: {end_time-start_time}" + '\n'
            log_stream += '\n'
            save_log(output_dic, switch_mode, log=log_stream)

def run_sft(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None
):
    dataset = get_dataset(model_args, data_args)
    model_name = model_args.model_name_or_path.lower()
    if 'llama' in model_name:
        block_prefix = llama_block_prefix
        model_name = 'llama'
    elif 'chatglm' in model_name:
        block_prefix = chatglm_block_prefix
        model_name = 'chatglm'
    elif 'bloom' in model_name:
        block_prefix = bloom_block_prefix
        model_name = 'bloom'
    else:
        raise ValueError(f"Invalid model type to create proper block prefix")

    if finetuning_args.debug_mode:
        dataset = dataset.select(range(1000))
        print("Debug mode is enabled. Only 1000 samples are used for training.")
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")

    model = model.to(torch.bfloat16) # The trainable parameters will be transformed into float 32 when performing update
    # model = model.to(torch.float32) # The trainable parameters will be transformed into float 32 when performing update

    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args_dict = training_args.to_dict()
    training_args_dict.update(dict(
        generation_max_length=training_args.generation_max_length or data_args.max_target_length,
        generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
    ))
    training_args = Seq2SeqTrainingArguments(**training_args_dict)

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = list(set([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids))
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    trainer = Seq2SeqPeftTrainer(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
        compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
        **split_dataset(dataset, data_args, training_args)
    )
    trainer.args.output_dir = f'./outputs/{model_name}/{data_args.dataset}/BAdam/{finetuning_args.switch_mode}'
    print(trainer.args.output_dir)
    # Training
    if training_args.do_train:
        # with torch.autocast(device_type=training_args.device.type):
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        # train_result = trainer.train(resume_from_checkpoint='./outputs/llama2-7b/centralized/checkpoint-4000/')
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        trainer.save_model()
        if trainer.is_world_process_zero() and model_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate: # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # # Predict
    # if training_args.do_predict:
    #     predict_results = trainer.predict(dataset, metric_key_prefix="predict", **gen_kwargs)
    #     if training_args.predict_with_generate: # predict_loss will be wrong if predict_with_generate is enabled
    #         predict_results.metrics.pop("predict_loss", None)
    #     trainer.log_metrics("predict", predict_results.metrics)
    #     trainer.save_metrics("predict", predict_results.metrics)
    #     trainer.save_predictions(predict_results)


# def run_adapter_sft(
#     model_args: "ModelArguments",
#     data_args: "DataArguments",
#     training_args: "Seq2SeqTrainingArguments",
#     finetuning_args: "FinetuningArguments",
#     generating_args: "GeneratingArguments",
#     callbacks: Optional[List["TrainerCallback"]] = None
# ):
#     dataset = get_dataset(model_args, data_args)
#
# #-----------------sgd-------------------
#     power_sgd = finetuning_args.power_sgd
#     if power_sgd:
#         training_args.optim = 'sgd'
#
#     if finetuning_args.debug_mode:
#         dataset = dataset.select(range(1000))
#         print("Debug mode is enabled. Only 1000 samples are used for training.")
#     model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args, training_args.do_train, stage="sft")
#
#     adapters.init(model)
#     if finetuning_args.finetuning_type == 'adap':
#         adapter_config = {'original_ln_before': True, 'original_ln_after': True, 'residual_before_ln': True,
#                           'adapter_residual_before_ln': False, 'ln_before': False, 'ln_after': False,
#                           'mh_adapter': False,
#                           'output_adapter': True, 'non_linearity': 'relu', 'reduction_factor': 32, 'inv_adapter': None,
#                           'inv_adapter_reduction_factor': None, 'cross_adapter': False, 'leave_out': []}
#
#         model.add_adapter("fedadapter", config=adapter_config)
#         model.train_adapter("fedadapter")
#         count_trainable(model)
#         model = model.to(torch.bfloat16) # The trainable parameters will be transformed into float 32 when performing update
#
#     elif finetuning_args.finetuning_type == 'adap_lora':
#         training_args.fp16 = True
#         config = LoRAConfig(r=16, alpha=32)
#         model.add_adapter("lora_adapter", config=config)
#         model.train_adapter("lora_adapter")
#         count_trainable(model)
#     else:
#         assert 1==0
#
#
#     # finetune config
#     num_clients = finetuning_args.num_clients
#     switch_mode = finetuning_args.switch_mode
#     bcd_epochs = finetuning_args.bcd_epochs
#
#     dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage="sft")
#     # len(dataset) = 52002
#
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
#     )
#
#     # Override the decoding parameters of Seq2SeqTrainer
#     training_args_dict = training_args.to_dict()
#     training_args_dict.update(dict(
#         generation_max_length=training_args.generation_max_length or data_args.max_target_length,
#         generation_num_beams=data_args.eval_num_beams or training_args.generation_num_beams
#     ))
#     training_args = Seq2SeqTrainingArguments(**training_args_dict)
#
#     split_dataset_dic = split_dataset(dataset, data_args, training_args)
#     total_train_dataset, total_eval_dataset = split_dataset_dic['train_dataset'], split_dataset_dic['eval_dataset']
#
#     # ########## FL dataset split begin###########
#     # total_train_dataset = total_train_dataset.select(range(0, 320)) # train_dataset for debug
#     # total_eval_dataset = total_eval_dataset.select(range(0, 24))    # eval_dataset for debug
#     par_data_num = len(total_train_dataset) // num_clients
#     clients_data = []
#     for i in range(num_clients):
#         start_idx = i * par_data_num
#         end_idx = start_idx + par_data_num if i < num_clients - 1 else len(total_train_dataset)
#         subset = total_train_dataset.select(range(start_idx, end_idx))
#         clients_data.append(subset)
#     # ########## FL dataset split end###########
#
#     # Keyword arguments for `model.generate`
#     gen_kwargs = generating_args.to_dict()
#     gen_kwargs["eos_token_id"] = list(set([tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids))
#     gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
#     gen_kwargs["logits_processor"] = get_logits_processor()
#
#     # For saving config
#     eval_results = []
#     output_dic = f'./outputs/llama2-7b/{data_args.dataset}/{num_clients}clients_{bcd_epochs}epochs_{switch_mode}_{training_args.optim}/'
#     output_path = output_dic + f'{switch_mode}.pkl'
#
#     if not os.path.exists(output_dic):
#         os.makedirs(output_dic)
#
#     if training_args.do_train:
#         update_flag = False
#         opt_flag = False
#         current_state_dict = None
#         log_stream = ""
#         # with torch.autocast(device_type=training_args.device.type):
#         for e in range(bcd_epochs):
#             start_time = time.time()
#             for cid in range(num_clients):
#                 client_model = copy.deepcopy(model)
#                 if update_flag:
#                     client_model.load_state_dict(current_state_dict)
#                 client_trainer = Seq2SeqPeftTrainer(
#                     finetuning_args=finetuning_args,
#                     client_id=cid,
#                     model=client_model,
#                     args=training_args,
#                     tokenizer=tokenizer,
#                     data_collator=data_collator,
#                     callbacks=callbacks,
#                     compute_metrics=ComputeMetrics(tokenizer) if training_args.predict_with_generate else None,
#                     # **split_dataset(clients_data[i], data_args, training_args)
#                     train_dataset=clients_data[cid],
#                     eval_dataset=total_eval_dataset,
#                 )
#                 client_trainer.args.output_dir = output_dic
#                 myckp = output_dic if opt_flag else None
#
#                 train_result = client_trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint, myckp=myckp)
#                 if (e + 1) % 3 == 0 and cid == num_clients - 1:
#                     client_trainer.save_state()
#                     client_trainer.save_model()
#
#                 if training_args.do_eval:
#                     metrics = client_trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
#                     if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
#                         metrics.pop("eval_loss", None)
#                     log_stream += outFunc(output_dic, round=e, cid=cid, bid=None, loss=metrics['eval_loss'], switch_mode=switch_mode)
#                     save_log(output_dic, switch_mode, log=log_stream)
#                     eval_results.append(metrics['eval_loss'])
#                 current_state_dict = client_model.state_dict()
#                 update_flag = True
#
#                 client_trainer._save_optimizer_and_scheduler(output_dir=output_dic)
#                 client_trainer._save_rng_state(output_dir=output_dic)
#                 opt_flag = True
#
#                 # free memory
#                 client_trainer._freegpu()
#                 del client_trainer
#                 del client_model
#                 del train_result
#                 gc.collect()
#                 torch.cuda.empty_cache()
#             end_time = time.time()
#             log_stream += f"{num_clients}clients, round{e}, {switch_mode}, round_time: {end_time-start_time}" + '\n'
#             log_stream += '\n'
#             save_log(output_dic, switch_mode, log=log_stream)
#         with open(output_path, 'wb') as f:
#             pickle.dump(eval_results, f)
#     # Evaluation
#
#

