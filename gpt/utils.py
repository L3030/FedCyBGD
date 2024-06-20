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
import math
import gc
import numpy as np
# import sys
# reload(sys)
# sys.setdefaultencoding( "utf-8" )
gpt2_block_list = [*[[f"transformer.h.{i}."] for i in range(12)], ["qa_outputs."]]

def get_block_param(model, block_id):
    return {name: param.detach().to('cpu') for name, param in model.named_parameters() if name.startswith(gpt2_block_list[block_id][0])}

def update_global(model, block_results):
    for k, v in block_results.items():
        if len(v) != 0:
            # k is the prefix_layer_name, v is the layer_state_dict()
            for name, param in model.named_parameters():
                if name.startswith(k):
                    if name in v:
                        param.data.copy_(v[name].to(model.device))

def aggregate_params(block_update):
    aggregated_params = {}
    num_updates = len(block_update)
    for name in block_update[0].keys():
        aggregated_params[name] = torch.zeros_like(block_update[0][name])
        for up in block_update:
            aggregated_params[name] += up[name]
        aggregated_params[name] /= num_updates
    return aggregated_params

def test_block_update(old, new, block):
    old_device = old.device
    new_device = new.device
    old.to('cpu')
    new.to('cpu')
    init_block_parameters = get_block_param(old, block)
    after_block_parameters = get_block_param(new, block)
    for k, v in after_block_parameters.items():
        difference_mask = init_block_parameters[k] != v
        break
    old.to(old_device)
    new.to(new_device)
    return difference_mask.sum().item()


def outFunc(output_dir, round, cid, bid, loss, switch_mode):
    assert switch_mode in ['fixed', 'random', 'agg', 'full', 'adap', 'lora']
    stream_log = ""
    stream_log += output_dir.split('/')[3] + '\n'
    if switch_mode in ['fixed', 'random', 'full']:
        assert isinstance(cid, int)
        if switch_mode == 'full':
            assert bid is None
            stream_log += f'Global training Round: {round}, client: {cid}, loss: {loss}' + '\n'
        else:
            stream_log += f'Global training Round: {round}, client: {cid}, block {bid}, loss: {loss}' + '\n'
    elif switch_mode in ['agg']:
        assert isinstance(cid, list)
        assert isinstance(loss, list)
        avg_loss = np.round(np.mean(loss[:-1]), decimals=5)
        agg_loss = np.round(loss[-1], decimals=5)
        stream_log += f'Global training Round: {round}, block {bid}' + '\n'
        stream_log += f'Client for current block: {cid}' + '\n'
        stream_log += f'Client loss for current block: {loss[:-1]}' + '\n'
        stream_log += f'Block {bid}‘s Average loss: {avg_loss}, Aggregation loss: {agg_loss}' + '\n'
    elif switch_mode in ['adap', 'lora']:
        stream_log += f'Global training Round: {round}, client_id {cid}' + '\n'
        stream_log += f'Client loss for current round: {loss}' + '\n'
    stream_log += '\n'

    return stream_log

def save_log(output_dir, switch_mode, log):
    file_name = output_dir + f'{switch_mode}' + '.log'
    fileObject = open(file_name, 'w', encoding='utf-8')
    fileObject.write(log)
    fileObject.close()

def count_trainable(model,):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (trainable_params / total_params) * 100
    print(
        f"trainable params: {trainable_params} || all params: {total_params} || trainable%: {trainable_percentage:.4f}")
    return trainable_params, total_params, trainable_percentage

def save_log(output_dir, log):
    if not os.path.exists('./outputs'):
        os.mkdir('./outputs')
    file_name = output_dir + '.log'
    fileObject = open(file_name, 'w', encoding='utf-8')
    fileObject.write(log)
    fileObject.close()