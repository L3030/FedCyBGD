from transformers import pipeline
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    LlamaForSequenceClassification
)
import adapters
import torch
import time
import numpy as np
import torch.nn.functional as F
from block_optim import BlockOptimizer, FLBlockOptimizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import os
os.environ["WANDB_DISABLED"] = "true"
from utils import *
import argparse
task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            start_forward = time.time()
            loss = self.compute_loss(model, inputs)
            end_forward = time.time()

        start_backward = time.time()
        self.accelerator.backward(loss)
        end_backward = time.time()



        forward_time = end_forward - start_forward
        backward_time = end_backward - start_backward

        print(f"Forward: {forward_time:.4f}, Backward: {backward_time:.4f}")

        return loss.detach() / self.args.gradient_accumulation_steps


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--load_pretrained_model", type=str, default=None)

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )

    # support enable_galore
    parser.add_argument("--enable_galore", action="store_true", help="Whether or not to use low rank optimizer.")
    # update_proj_gap
    parser.add_argument("--update_proj_gap", type=int, default=50)
    # galore_scale
    parser.add_argument("--galore_scale", type=float, default=1.0)
    # proj_type
    parser.add_argument("--proj_type", type=str, default="std")
    # lora_all_modules
    parser.add_argument("--lora_all_modules", action="store_true", help="Whether or not to use lora for all modules.")
    # eval_llama
    parser.add_argument("--eval_llama", action="store_true", help="Whether or not to evaluate llama model.")
    # low_rank_method
    parser.add_argument("--low_rank_method", type=str, default=None, help="low rank method for wandb sweep")

    # blockoptimizer config
    parser.add_argument("--switch_block_every", type=int, default=20)
    parser.add_argument("--switch_mode", type=str, default="fixed")
    parser.add_argument("--power_sgd", action="store_true", help="Activate power_sgd")
    parser.add_argument("--block_type", choices=['fl', 'badam'])
    parser.add_argument("--lora", action="store_true", help="Activate lora")
    parser.add_argument("--full", action="store_true", help="Activate full-tuning")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--num_clients", type=int, default=22)

    args = parser.parse_args()
    return args


def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

def create_optimizer(model, args, client_step):
    before_memory = torch.cuda.memory_allocated('cuda:0')
    print(f"Memory allocated before operation: {before_memory / 1024 ** 2:.2f} MB")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # for fl block learning
    # client_step = len(train_dataloader) // args.num_clients // args.gradient_accumulation_steps
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-6)

    # optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.full:
        pass
    else:
        if not args.lora:
            if args.block_type in ['fl', 'badam']:
                assert args.model_name_or_path in ['roberta-base', 'roberta-large'], 'Block_prefix_list should be predefined here.'
                block_prefix_list = args.model_name_or_path
                if args.block_type == 'fl':
                    optimizer = FLBlockOptimizer(base_optimizer=optimizer,
                                                 named_parameters_list=list(model.named_parameters()),
                                                 parameters_list=list(model.parameters()),
                                                 block_prefix_list=block_prefix_list,
                                                 switch_block_every=args.switch_block_every,
                                                 switch_mode='random',
                                                 log_fn=None,
                                                 start_block=None,
                                                 verbose=0,
                                                 power_sgd=args.power_sgd,
                                                 client_step=client_step
                                                 )
                else:
                    optimizer = BlockOptimizer(base_optimizer=optimizer,
                                               named_parameters_list=list(model.named_parameters()),
                                               block_prefix_list=block_prefix_list,
                                               switch_block_every=args.switch_block_every,
                                               switch_mode='ascending',
                                               log_fn=None,
                                               start_block=None,
                                               verbose=1,
                                               )

        else:
            adapters.init(model)
            adapter_config = {'original_ln_before': True, 'original_ln_after': True, 'residual_before_ln': True,
                              'adapter_residual_before_ln': False, 'ln_before': False, 'ln_after': False,
                              'mh_adapter': False,
                              'output_adapter': True, 'non_linearity': 'relu', 'reduction_factor': 32,
                              'inv_adapter': None,
                              'inv_adapter_reduction_factor': None, 'cross_adapter': False, 'leave_out': []}
            model.add_adapter("fedadapter", config=adapter_config)

            # model.add_adapter("fedadapter", config="lora")
            model.train_adapter("fedadapter")
    return model, optimizer


if __name__ == '__main__':
    # unmasker = pipeline('fill-mask', model='roberta-base')
    # results = unmasker(["The man worked as a <mask>.", "The woman worked as a <mask>."])
    # # classifier = pipeline("sentiment-analysis")
    # # results = classifier(["we are very happy", "we hope you don't hate it"])
    # #
    # for result in results:
    #     print(result)
    # raw_dataset = load_dataset("glue", "mrpc")
    # print(raw_dataset)
    # tokeniszer = AutoTokenizer.from_pretrained('roberta-base')
    # input_ids = tokeniszer('this is a sentence', 'this is another sentence')
    # print(input_ids)
    # GLUE_TASKS = [ "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
    args = parse_args()

    GLUE_TASKS = ["cola"]
    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True,)

    for e in range(args.epochs):
        task = args.task_name
        assert task in task_to_keys.keys()

        actual_task = "mnli" if task == "mnli-mm" else task
        dataset = load_dataset('glue', actual_task, )
        metric = load_metric('glue', actual_task, )
        sentence1_key, sentence2_key = task_to_keys[task]
        encoder_dataset = dataset.map(preprocess_function, batched=True)
        total_train_num = len(encoder_dataset['train'])

        client_train_num = total_train_num // args.num_clients
        client_step = client_train_num // args.per_device_train_batch_size
        # print(dataset)
        #     encoder_train_dataset = dataset['train'].map(preprocess_function, batched=True)
        #     encoder_val_dataset = dataset['validation'].map(preprocess_function, batched=True)
        #     encoder_test_dataset = dataset['test'].map(preprocess_function, batched=True)
        num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
        # print(encoder_dataset)
        # print(encoder_dataset['train'])
        # print(encoder_dataset['test'])
        # print(encoder_dataset['train'][0])
        # print(encoder_dataset['test'][0])

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels,)
        model, optimizer = create_optimizer(model, args, client_step)

        metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

        train_args = TrainingArguments(
            "./"+task,
            overwrite_output_dir=True,
            evaluation_strategy = 'steps',
            eval_steps = total_train_num//args.per_device_train_batch_size//20,
            save_strategy= 'no',
            do_predict=True,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            weight_decay=0.01,
            load_best_model_at_end=False,
            metric_for_best_model= metric_name,
            gradient_accumulation_steps=1,
            logging_steps=10
        )
        validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
        trainer = CustomTrainer(
            model,
            train_args,
            # train_dataset=encoder_train_dataset,
            # eval_dataset=encoder_val_dataset,
            train_dataset= encoder_dataset['train'],
            eval_dataset= encoder_dataset[validation_key],
            tokenizer= tokenizer,
            compute_metrics= compute_metrics,
            # block=True#args.block
            optimizers=(optimizer, None)
        )
        count_trainable(model)
        bg = time.time()
        trainer.train()
        ed = time.time()

    # print(f"take {ed - bg} seconds to train")
    # print(trainer.evaluate())
    # print(trainer.predict(encoder_test_dataset))
    # print(trainer.predict(encoder_dataset['test'].map(preprocess_function, batched=True)))