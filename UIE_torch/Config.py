
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
import configparser
import torch
con = configparser.ConfigParser()
class Config():
    def __init__(self,config_file):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        file = config_file
        con.read(file,encoding='utf8')
        items = con.items('path')
        path = dict(items)
        items = con.items('model_superparameter')
        model_sp = dict(items)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.gpus = [0]
            torch.cuda.set_device('cuda:{}'.format(self.gpus[0]))
        self.model_path = path['model_path']
        self.log_path = path['log_path']
        self.model_save_path = path['model_save_path']
        self.out_put_dir = path['out_put_dir']
        self.multi_task_config = path['multi_task_config']
        self.metric_for_best_model = model_sp['metric_for_best_model']
        self.epochs = eval(model_sp['epochs'])
        self.tr_loss = eval(model_sp['tr_loss'])
        self.logging_loss = eval(model_sp['logging_loss'])
        self.global_steps = eval(model_sp['global_steps'])
        self.max_target_length = eval(model_sp['max_target_length'])
        self.logging_steps = eval(model_sp['logging_steps'])
        self.learning_rate = eval(model_sp['lr'])
        self.warmup_ratio = eval(model_sp['warmup_ratio'])
        self.writer_type = model_sp['writer_type']
        self.max_source_length = eval(model_sp['max_source_length'])
        self.max_prefix_length = None
        self.max_target_length = eval(model_sp['max_target_length'])
        self.negative_keep = eval(model_sp['negative_keep'])
        self.ignore_pad_token_for_loss = True
        self.meta_positive_rate = eval(model_sp['negative_keep'])
        self.meta_negative = -1
        self.ordered_prompt  = False
        self.batch_size = eval(model_sp['batch_size'])

        