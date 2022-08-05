#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chaoyi Yuan, chaoyiyuan3721@gmail.com
# Status: Active
from pickle import FALSE
from typing import List
import json
import random
import os
import logging
import tabulate
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset,DataLoader
from typing import Optional
from uie.seq2struct.utils_dataset import load_dataset
from uie.seq2struct.data_collator_torch import UIEDataset,DynamicSSIGenerator
from uie.evaluation import constants
from uie.evaluation.sel2record import (
    SEL2Record,
    RecordSchema,
    MapConfig,
    merge_schema,
)
from uie.seq2struct.t5tokenizer import T5BertTokenizer
from Config import Config
# config_path = 'config.ini'
# config = Config(config_path)
# max_source_length = config.max_source_length
# max_prefix_length = config.max_prefix_length
# max_target_length = config.max_target_length
# negative_keep = config.negative_keep
# multi_task_config = config.multi_task_config
# ignore_pad_token_for_loss = config.ignore_pad_token_for_loss
# meta_positive_rate = config.meta_positive_rate
# meta_negative = config.meta_negative
# ordered_prompt  = config.ordered_prompt
# batch_size = config.batch_size
max_source_length = 384
max_prefix_length = None
max_target_length = 192
negative_keep = 1.0
multi_task_config = './config/multi-task-duuie.yaml',
ignore_pad_token_for_loss = True
meta_positive_rate = 1
meta_negative = -1
ordered_prompt  = False
batch_size = 16
logger = logging.getLogger("__main__")
def set_logger(output_dir):
    """ Set logger """
    logger.setLevel(logging.DEBUG if 'DEBUG' in os.environ else logging.INFO)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(
                filename=f"{output_dir}.log",
                mode="w",
                encoding="utf-8",
            )
        ],
    )
    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level=logging.DEBUG)
    # add formatter to console_handler
    console_handler.setFormatter(fmt=logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # add console_handler to logger
    logger.addHandler(console_handler)
def get_writer(logging_dir,writer_type):
    if writer_type == "visualdl":
        from visualdl import LogWriter
        writer = LogWriter(logdir=logging_dir)
    elif writer_type == "tensorboard":
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=logging_dir)
    else:
        raise ValueError("writer_type must be in ['visualdl', 'tensorboard']")
    return writer
def read_json_file(file_name):
    """ Read jsonline file as generator """
    with open(file_name, encoding='utf8') as fin:
        for line in fin:
            yield json.loads(line)


def better_print_multi(results):
    """ Better print multi task results
    results: Dictionary of task and metric {"task:metric": "value", ...}
    """
    table = [(task, results[task]) for task in results]
    return tabulate.tabulate(table, headers=['Task', 'Metric'])


def read_func(tokenizer,
              data_file: str,
              max_source_length: int,
              is_train: bool = False,
              negative_keep: float = 1.0):
    """ Read instance from data_file

    Args:
        tokenizer (PretrainedTokenizer): Tokenizer
        data_file (str): Data filename
        max_source_length (int): Max source length
        is_train (bool): instance from this file whether for training
        negative_keep (float): the ratio of keeping negative instances
    """

    negative_drop_num = 0
    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance = json.loads(line)

            # Drop negative sample in random during training stage
            if is_train and len(instance['spot_asoc']) == 0:
                # if negative_keep >= 1, keep all negative instances
                # else drop negative instance when random() > negative_keep
                if random.random() > negative_keep:
                    negative_drop_num += 1
                    continue

            inputs = tokenizer(
                instance['text'],
                return_token_type_ids=False,
                return_attention_mask=True,
                max_seq_len=max_source_length,
                truncation=True
            )

            # `sample_ssi` can be True in the training stage
            # `sample_ssi` can only be False in the evaluation stage
            # 在训练时，ssi可以动态变化 (sample_ssi=True)
            # 但是在推理和验证时，ssi必须固定保证推理结果的一致 (sample_ssi=False)
            inputs.update({
                'spots': instance['spot'],
                'asocs': instance['asoc'],
                'spot_asoc': instance['spot_asoc'],
                'sample_ssi': is_train
            })
            yield inputs

    if negative_drop_num > 0:
        logger.info(
            f'Drop negative {negative_drop_num} instance during loading {data_file}.'
        )


def read_training_instance_based_config(tokenizer,
                                        config_file: str,
                                        max_source_length: int,
                                        negative_keep: float = 1.0):
    """Read training instances based on config_file

    Args:
        tokenizer (PretrainedTokenizer): Tokenizer
        config_file (str): Config filename
        max_source_length (int): Max source length
        negative_keep: the ratio of keeping negative instances

    Yields:
        dict: instance for training
    """
    task_configs = list(TaskConfig.load_list_from_yaml(config_file))

    for task_config in task_configs:
        negative_drop_num = 0

        train_file = os.path.join(task_config.data_path, "train.json")
        schema_file = os.path.join(task_config.data_path, "record.schema")
        record_schema = RecordSchema.read_from_file(schema_file)
        with open(train_file, 'r', encoding='utf-8') as fin:
            count = 0
            for line in fin:
                instance = json.loads(line)

                # Drop negative sample in random during training stage
                if len(instance['spot_asoc']) == 0:
                    # if negative_keep >= 1, keep all negative instances
                    # else drop negative instance when random() > negative_keep
                    if random.random() > negative_keep:
                        negative_drop_num += 1
                        continue

                inputs = tokenizer(
                    instance['text'],
                    return_token_type_ids=False,
                    return_attention_mask=True,
                    max_length=max_source_length,
                    truncation=True)
                # `sample_ssi` is True in the training stage
                inputs.update({
                    'spots': record_schema.type_list,
                    'asocs': record_schema.role_list,
                    'spot_asoc': instance['spot_asoc'],
                    'sample_ssi': True
                })
                yield inputs
                count += 1
            logger.info(f"Load {count} instances from {train_file}")

        if negative_drop_num > 0:
            logger.info(
                f'Drop negative {negative_drop_num} instance during loading {train_file}.'
            )
   # Merge schema in all datasets for pre-tokenize

def get_train_dataloader(model, tokenizer):

    # logger.info(f'Load data according to {args.multi_task_config} ...')
    dataset = load_dataset(read_training_instance_based_config, #tokenizer格式但ssi没有
                           tokenizer=tokenizer,
                           config_file=multi_task_config,
                           max_source_length=max_source_length,
                           lazy=False,
                           negative_keep=negative_keep)

    # Merge schema in all datasets for pre-tokenize
    schema_list = list()
    for task_config in TaskConfig.load_list_from_yaml(multi_task_config): #读取schema
        schema_file = os.path.join(task_config.data_path, "record.schema")
        schema_list += [RecordSchema.read_from_file(schema_file)]
    schema = merge_schema(schema_list)
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    train_data = UIEDataset(
        dataset,
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        max_source_length=max_source_length,
        max_prefix_length=max_prefix_length,
        max_target_length=max_target_length,
        ssi_generator=DynamicSSIGenerator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=meta_positive_rate,
            negative= meta_negative,
            ordered_prompt=ordered_prompt,
        ),
        spot_asoc_nosier=None,
    )

    # collate_fn = DataCollatorForMultiTaskSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     max_source_length=max_source_length,
    #     max_prefix_length=max_prefix_length,
    #     max_target_length=max_target_length,
    #     ssi_generator=DynamicSSIGenerator(
    #         tokenizer=tokenizer,
    #         schema=schema,
    #         positive_rate=meta_positive_rate,
    #         negative=meta_negative,
    #         ordered_prompt=ordered_prompt,
    #     ),
    #     # spot_asoc_nosier=spot_asoc_nosier,
    # )
    # train_data = Dataset(dataset)
    train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
    # train_dataloader = DataLoader(dataset=train_data,
    #                          collate_fn=collate_fn,
    #                          shuffle=True),
    return train_dataloader

def read_func(tokenizer,
              data_file: str,
              max_source_length: int,
              is_train: bool = False,
              negative_keep: float = 1.0):
    """ Read instance from data_file

    Args:
        tokenizer (PretrainedTokenizer): Tokenizer
        data_file (str): Data filename
        max_source_length (int): Max source length
        is_train (bool): instance from this file whether for training
        negative_keep (float): the ratio of keeping negative instances
    """

    negative_drop_num = 0
    with open(data_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance = json.loads(line)

            # Drop negative sample in random during training stage
            if is_train and len(instance['spot_asoc']) == 0:
                # if negative_keep >= 1, keep all negative instances
                # else drop negative instance when random() > negative_keep
                if random.random() > negative_keep:
                    negative_drop_num += 1
                    continue

            inputs = tokenizer(
                instance['text'],
                return_token_type_ids=False,
                return_attention_mask=True,
                max_length=max_source_length,
                truncation=True
            )

            # `sample_ssi` can be True in the training stage
            # `sample_ssi` can only be False in the evaluation stage
            # 在训练时，ssi可以动态变化 (sample_ssi=True)
            # 但是在推理和验证时，ssi必须固定保证推理结果的一致 (sample_ssi=False)
            inputs.update({
                'spots': instance['spot'],
                'asocs': instance['asoc'],
                'spot_asoc': instance['spot_asoc'],
                'sample_ssi': is_train
            })
            yield inputs

    if negative_drop_num > 0:
        logger.info(
            f'Drop negative {negative_drop_num} instance during loading {data_file}.'
        )
def get_eval_dataloader(model, tokenizer, eval_filename, record_schema, max_source_length):
    """ Get evaluation dataloader
    """

    logger.info(f'Load data from {eval_filename} ...')

    schema = RecordSchema.read_from_file(record_schema)

    # dataset = load_dataset(read_func,
    #                        tokenizer=tokenizer,
    #                        data_file=eval_filename,
    #                        max_source_length=args.max_source_length,
    #                        is_train=False,
    #                        lazy=False)
    dataset = load_dataset(read_func, #tokenizer格式但ssi没有
                           tokenizer=tokenizer,
                           data_file = eval_filename,
                           max_source_length=max_source_length,
                           is_train=False,
                           lazy=False)

    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
    train_data = UIEDataset(
        dataset,
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        max_source_length=max_source_length,
        max_prefix_length=max_prefix_length,
        max_target_length=max_target_length,
        ssi_generator=DynamicSSIGenerator(
            tokenizer=tokenizer,
            schema=schema,
            positive_rate=meta_positive_rate,
            negative= meta_negative,
            ordered_prompt=ordered_prompt,
        ),
        spot_asoc_nosier=None,
    )
    # collate_fn = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     model=model,
    #     label_pad_token_id=label_pad_token_id,
    #     max_source_length=max_source_length,
    #     max_prefix_length=max_prefix_length,
    #     max_target_length=max_target_length,
    #     ssi_generator=DynamicSSIGenerator(
    #         tokenizer=tokenizer,
    #         schema=schema,
    #         positive_rate=1,
    #         negative=-1,
    #         ordered_prompt=True,
    #     ),
    # )
    dev_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=False)

    return dev_dataloader
def better_print_multi(results):
    """ Better print multi task results
    results: Dictionary of task and metric {"task:metric": "value", ...}
    """
    table = [(task, results[task]) for task in results]
    return tabulate.tabulate(table, headers=['Task', 'Metric'])

def load_eval_tasks(model, tokenizer):
    """ Load evaluation tasks

    Args:
        model (PretrainedModel): Pretrain Model
        tokenizer (PretrainedTokenizer): Tokenizer
        args (Namespace): arguments for loading eval tasks

    Returns:
        list(Task): list of evaluation tasks
    """
    eval_tasks = dict()
    task_configs = list(TaskConfig.load_list_from_yaml(multi_task_config))

    for task_config in task_configs:

        val_filename = os.path.join(task_config.data_path, 'val.json')
        record_schema = os.path.join(task_config.data_path, 'record.schema')

        task_dataloader = get_eval_dataloader(model=model,
                                              tokenizer=tokenizer,
                                              eval_filename=val_filename,
                                              record_schema=record_schema,
                                              max_source_length= max_source_length
                                              )

        sel2record = SEL2Record(
            schema_dict=SEL2Record.load_schema_dict(task_config.data_path),
            map_config=MapConfig.load_by_name(task_config.sel2record),
            tokenizer=tokenizer
            if isinstance(tokenizer, T5BertTokenizer) else None,
        )

        eval_tasks[task_config.dataset_name] = Task(
            config=task_config,
            dataloader=task_dataloader,
            sel2record=sel2record,
            val_instances=list(read_json_file(val_filename)),
            metrics=task_config.metrics,
        )

    return eval_tasks


# def write_prediction(eval_prediction, output_dir, prefix='eval'):
#     """Write prediction to output_dir

#     Args:
#         eval_prediction (dict):
#             - `record` (list(dict)), each element is extraction reocrd
#             - `sel` (list(str)): each element is sel expression
#             - `metric` (dict)
#         output_dir (str): Output directory path
#         prefix (str, optional): prediction file prefix. Defaults to 'eval'.

#     Write prediction to files:
#         - `preds_record.txt`, each line is extracted record
#         - `preds_seq2seq.txt`, each line is generated sel
#         - `results.txt`, detailed metrics of prediction
#     """
#     output_filename = os.path.join(output_dir, f"{prefix}-preds_record.txt")
#     with open(output_filename, 'w', encoding='utf8') as output:
#         for pred in eval_prediction.get('record', []):
#             output.write(json.dumps(pred, ensure_ascii=False) + '\n')

#     output_filename = os.path.join(output_dir, f"{prefix}-preds_seq2seq.txt")
#     with open(output_filename, 'w', encoding='utf8') as output:
#         for pred in eval_prediction.get('sel', []):
#             output.write(pred + '\n')

#     output_filename = os.path.join(output_dir, f"{prefix}-results.txt")
#     with open(output_filename, 'w', encoding='utf8') as output:
#         for key, value in eval_prediction.get('metric', {}).items():
#             output.write(f"{prefix}-{key} = {value}\n")


class TaskConfig:
    def __init__(self, task_dict) -> None:
        self.dataset_name = task_dict.get('name', '')
        self.task_name = task_dict.get('task', '')
        self.data_path = task_dict.get('path', '')
        self.sel2record = task_dict.get('sel2record', '')
        self.metrics = task_dict.get('metrics', [])
        self.eval_match_mode = task_dict.get('eval_match_mode', 'normal')
        self.schema = RecordSchema.read_from_file(
            f"{self.data_path}/record.schema")

    def __repr__(self) -> str:
        task_config_list = [
            f"dataset: {self.dataset_name}", f"task   : {self.task_name}",
            f"path   : {self.data_path}", f"schema : {self.schema}",
            f"metrics: {self.metrics}",
            f"eval_match_mode : {self.eval_match_mode}"
        ]
        return '\n'.join(task_config_list)

    @staticmethod
    def load_list_from_yaml(task_config):
        import yaml
        configs = yaml.load(open(task_config[0], encoding='utf8'),
                            Loader=yaml.FullLoader)
        task_configs = filter(lambda x: x.startswith('T'), configs)
        for task_config in task_configs:
            yield TaskConfig(configs[task_config])


@dataclass
class Task:
    config: TaskConfig
    dataloader: DataLoader
    sel2record: SEL2Record
    val_instances: List[dict]
    metrics: List[str]
