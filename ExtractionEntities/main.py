import re
import json
import os
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="biaffine", choices=['biaffine', 'UnlabeledEntity', 'globalpointer'], help='choice model type')
parser.add_argument('--config_file', type=str, default="./train_config/config_yang.ini",  help='choice config file')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel, BertConfig,AdamW,BertTokenizerFast,get_linear_schedule_with_warmup
from data_processing.data_process import yeild_data
from loss_function.loss_fun import multilabel_categorical_crossentropy,global_pointer_crossentropy
from metrics.metrics import global_pointer_f1_score
from torch.nn.utils import clip_grad_norm_
from Config import Config
config = Config(args.config_file,args.model_type)
from utils.Logginger import init_logger
logger = init_logger("ner", logging_path=config.log_path)
tokenizer = BertTokenizerFast.from_pretrained(config.model_path,do_lower_case= True)

train_dataloader,categories_size,categories2id,_= yeild_data(
    config.train_file_data,
    is_train=True,
    maxlen=config.maxlen,
    batch_size=config.batch_size,
    tokenizer=tokenizer,
    DDP=False
)
val_dataloader = yeild_data(
    config.val_file_data,
    is_train=False,
    maxlen=config.maxlen,
    batch_size=config.batch_size,
    tokenizer=tokenizer,
    categories_size=categories_size,
    categories2id=categories2id,
    DDP=False
)

if config.model_type == "biaffine":
    from model.model import BiaffineNet
    model = BiaffineNet(
        config.model_path,
        categories_size,
        hidden_size=config.hidden_size,
        dim_in = config.dim_in,
        dim_hid = config.dim_hid,
        abPosition=config.abPosition
    )
elif config.model_type=="UnlabeledEntity":
    from model.model import UnlabeledEntityNet
    model = UnlabeledEntityNet(
        config.model_path,
        categories_size,
        hidden_size=config.hidden_size,
        abPosition=config.abPosition,
        rePosition=config.rePosition,
        re_maxlen= config.re_maxlen,
        max_relative=config.max_relative
    )
elif config.model_type=="globalpointer":
    from model.model import GlobalPointerNet
    model = GlobalPointerNet(
        config.model_path,
        categories_size,
        config.head_size,
        config.hidden_size
    ).to(config.device)


if config.use_gpu:
    model = nn.DataParallel(
        model.to(config.device), 
        device_ids=config.gpus, 
        output_device=config.gpus[0]
    )

total_steps = len(train_dataloader) * config.epochs
param_optimizer = list(model.named_parameters())

optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in config.no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in config.no_decay)],
        'weight_decay_rate': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config.warmup_steps,
    num_training_steps=total_steps
)

def train(dataloader, model, loss_func, optimizer):
    model.train()
    size = len(dataloader.dataset)
    numerate, denominator = 0, 0
    for batch, (data,label) in enumerate(dataloader):
        input_ids = data['input_ids'].squeeze().to(config.device)
        attention_mask = data['attention_mask'].squeeze().to(config.device)
        token_type_ids = data['token_type_ids'].squeeze().to(config.device)
        label = label.to(config.device)
        pred = model(input_ids,attention_mask,token_type_ids)
        loss = loss_func(label,pred)
        temp_n,temp_d = global_pointer_f1_score(label,pred)
        numerate += temp_n
        denominator += temp_d
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    logger.info(f"Train F1: {(2*numerate/denominator):>4f}%")

def evaluate(dataloader,loss_func, model):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0
    numerate, denominator = 0, 0
    with torch.no_grad():
        for data,label in dataloader:
            input_ids = data['input_ids'].squeeze().to(config.device)
            attention_mask = data['attention_mask'].squeeze().to(config.device)
            token_type_ids = data['token_type_ids'].squeeze().to(config.device)
            label = label.squeeze().to(config.device)
            pred = model(input_ids, attention_mask, token_type_ids)
            val_loss += loss_func(label,pred).item()
            temp_n,temp_d = global_pointer_f1_score(label,pred)
            numerate += temp_n
            denominator += temp_d
    val_loss /= size
    val_f1 = 2*numerate/denominator
    logger.info(f"Test Error: \n ,F1:{(val_f1):>4f},Avg loss: {val_loss:>8f} \n")
    return val_f1

class Evaluator(object):
    """评估与保存
    """
    def __init__(self,best_val_f1):
        self.best_val_f1 = best_val_f1
    def on_epoch_end(self, epoch, logs=None):
        f1 = evaluate(val_dataloader,global_pointer_crossentropy, model)
        # f1, precision, recall = evaluate_val(val_file_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            torch.save(model.state_dict(), f=f"{config.model_save_path}_{config.model_type}.pth")
        logger.info(
            'valid:  f1: %.5f,  best f1: %.5f\n' %
            (f1,self.best_val_f1)
            # 'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            # (f1, precision, recall, self.best_val_f1)
        )
        return self.best_val_f1

def run_model(optimizer):
    best_val_f1 = 0
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}")
        train(train_dataloader, model, global_pointer_crossentropy, optimizer)
        best_val_f1 = Evaluator(best_val_f1).on_epoch_end(epoch)
    logger.info('end')

if __name__ == '__main__':
    run_model(optimizer)