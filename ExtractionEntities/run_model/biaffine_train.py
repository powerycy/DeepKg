import re
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel, BertConfig,AdamW,BertTokenizerFast,get_linear_schedule_with_warmup
from data_processing.data_process import yeild_data
from model.model import BiaffineNet
from loss_function.loss_fun import multilabel_categorical_crossentropy,global_pointer_crossentropy
from metrics.metrics import global_pointer_f1_score
import sys
from torch.nn.utils import clip_grad_norm_
gpus = [3,5,6]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("Using {} device".format(device))
import configparser
con = configparser.ConfigParser()
file = './train_config/config.ini'
con.read(file,encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
model_path = path['model_path']
train_file_data = path['train_file_data']
val_file_data = path['val_file_data']
model_save_path = path['model_save_path']
head_size = eval(model_sp['head_size'])
hidden_size = eval(model_sp['hidden_size'])
learning_rate = eval(model_sp['learning_rate'])
clip_norm = eval(model_sp['clip_norm'])
abPosition = eval(model_sp['abposition'])
dim_in = eval(model_sp['dim_in'])
dim_hid = eval(model_sp['dim_hid'])
train_dataloader,categories_size,categories2id,_= yeild_data(train_file_data,is_train=True,DDP=False)
val_dataloader = yeild_data(val_file_data,categories_size=categories_size,categories2id=categories2id,is_train=False,DDP=False)
model = BiaffineNet(model_path,categories_size,hidden_size=hidden_size,dim_in = dim_in,dim_hid = dim_hid,abPosition=abPosition)
model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
epochs = eval(model_sp['epochs'])
warmup_steps = eval(model_sp['warmup_steps'])
total_steps = len(train_dataloader) * epochs
param_optimizer = list(model.named_parameters())
# train_epoch_loss = 0
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
def train(dataloader, model, loss_func, optimizer):
    model.train()
    size = len(dataloader.dataset)
    numerate, denominator = 0, 0
    for batch, (data,label) in enumerate(dataloader):
        input_ids = data['input_ids'].squeeze().to(device)
        attention_mask = data['attention_mask'].squeeze().to(device)
        token_type_ids = data['token_type_ids'].squeeze().to(device)
        label = label.to(device)
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Train F1: {(2*numerate/denominator):>4f}%")

def evaluate(dataloader,loss_func, model):
    size = len(dataloader.dataset)
    model.eval()
    val_loss = 0
    numerate, denominator = 0, 0
    with torch.no_grad():
        for data,label in dataloader:
            input_ids = data['input_ids'].squeeze().to(device)
            attention_mask = data['attention_mask'].squeeze().to(device)
            token_type_ids = data['token_type_ids'].squeeze().to(device)
            label = label.squeeze().to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            val_loss += loss_func(label,pred).item()
            temp_n,temp_d = global_pointer_f1_score(label,pred)
            numerate += temp_n
            denominator += temp_d
    val_loss /= size
    val_f1 = 2*numerate/denominator
    print(f"Test Error: \n ,F1:{(val_f1):>4f},Avg loss: {val_loss:>8f} \n")
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
            torch.save(model.state_dict(), f=model_save_path)
        print(
            'valid:  f1: %.5f,  best f1: %.5f\n' %
            (f1,self.best_val_f1)
            # 'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            # (f1, precision, recall, self.best_val_f1)
        )
        return self.best_val_f1
def run_model(optimizer):
    best_val_f1 = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train(train_dataloader, model, global_pointer_crossentropy, optimizer)
        best_val_f1 = Evaluator(best_val_f1).on_epoch_end(epoch)
    print('end')
if __name__ == '__main__':
    run_model(optimizer)