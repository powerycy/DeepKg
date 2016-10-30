# -*- encoding: utf-8 -*-

import random
from typing import List
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import jsonlines
import numpy as np
import torch
import re
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
# from scipy.stats import spearmanr
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer
import faiss
import faiss.contrib.torch_utils
# 基本参数
EPOCHS = 5
SAMPLES = 10000
BATCH_SIZE = 64
LR = 3e-5
DROPOUT = 0.3
MAXLEN = 64
POOLING = 'cls'   # choose in ['cls', 'pooler', 'first-last-avg', 'last-avg']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# 预训练模型目录
# BERT = 'pretrained_model/bert_pytorch'
# BERT_WWM_EXT = 'pretrained_model/bert_wwm_ext_pytorch'
# ROBERTA = 'pretrained_model/roberta_wwm_ext_pytorch'
model_path='/home/yuanchaoyi/BeiKe/QA_match/roberta_base'

# 微调后参数存放位置
SAVE_PATH = './saved_model/simcse_unsup.pt'

# 数据目录
train_file_data_pos = './train_clean'
train_file_data_neg = './train_dirty'
test_data = './test_all'
def build_index(vecs,ids,nlist=256):
    dim = vecs.shape[1]
    # quant = faiss.IndexFlatIP(dim)
    # index = faiss.IndexIVFFlat(quant, dim, min(nlist, vecs.shape[0]))
    res = faiss.StandardGpuResources()
    res.noTempMemory()
    index = faiss.GpuIndexIVFFlat(res, dim, min(nlist, vecs.shape[0]), faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add_with_ids(vecs,ids)
    return index
def encode_batch(texts,model):
    with torch.no_grad():
        text_encs = tokenizer(texts,padding=True,
                                    max_length=MAXLEN,
                                    truncation=True,
                                    return_tensors="pt")
        input_ids = text_encs["input_ids"].to(DEVICE)
        attention_mask = text_encs["attention_mask"].to(DEVICE)
        token_type_ids = text_encs["token_type_ids"].to(DEVICE)
        output = model(input_ids, attention_mask, token_type_ids)
    return output
def sim_query(index,sentence,model,id2text,topK=20):
    vec = encode_batch([sentence],model)
    vec = vec / vec.norm(dim=1, keepdim=True)
    # vec = vec.cpu().numpy()
    _, sim_idx = index.search(vec, topK)
    sim_sentences = []
    for i in range(sim_idx.shape[1]):
        idx = sim_idx[0, i]
        if idx.item() == -1:
            continue
        sim_sentences.append(id2text[idx.item()])
    return sim_sentences
def encode_file(fname,model):
    all_texts = []
    all_ids = []
    all_vecs = []
    with open(fname, "r", encoding="utf8") as h:
        texts = []
        idxs = []
        for idx, line in tqdm(enumerate(h)):
            if not line.strip():
                continue
            # line = line.split('||')
            texts.append(line.split('\t')[0].strip())
            idxs.append(idx)
            if len(texts) >= BATCH_SIZE:
                vecs = encode_batch(texts,model)
                vecs = vecs / vecs.norm(dim=1, keepdim=True)
                all_texts.extend(texts)
                all_ids.extend(idxs)
                all_vecs.append(vecs.cpu())
                texts = []
                idxs = []
    all_vecs = torch.cat(all_vecs, 0)
    id2text = {idx: text for idx, text in zip(all_ids, all_texts)}
    all_ids= torch.tensor(all_ids, dtype=torch.int64)
    return id2text,all_vecs,all_ids
def load_data(file,file2):
    with open(file) as train_pos:
        with open(file2) as train_neg:
            D = []
            D_dev = []
            for num,line in enumerate(train_pos):
                line = re.sub('\n','',line)
                if num % 7 == 0:
                    D_dev.append(line)
                else:
                    D.append(line)
            for num,line in enumerate(train_neg):
                line = re.sub('\n','',line)
                if num % 7 == 0:
                    D_dev.append(line)
                else:
                    D.append(line)
    return D,D_dev

class TrainDataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
      
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        # 添加自身两次, 经过bert编码之后, 互为正样本
        return tokenizer([text, text], max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        return self.text_2_id(self.data[index])
    

class TestDataset(Dataset):
    """测试数据集, 重写__getitem__和__len__方法"""
    def __init__(self, data: List):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def text_2_id(self, text: str):
        return tokenizer(text, max_length=MAXLEN, truncation=True, padding='max_length', return_tensors='pt')
    
    def __getitem__(self, index: int):
        da = self.data[index]        
        return self.text_2_id([da[0]]), self.text_2_id([da[1]]), int(da[2])


class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""
    def __init__(self, pretrained_model, pooling):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)       
        config.attention_probs_dropout_prob = DROPOUT   # 修改config的dropout系数
        config.hidden_dropout_prob = DROPOUT           
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling
        
    def forward(self, input_ids, attention_mask, token_type_ids):

        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        
        if self.pooling == 'pooler':
            return out.pooler_output            # [batch, 768]
        
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)    # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)       # [batch, 768]
        
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)    # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)    # [batch, 768, seqlen]                   
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1) # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)   # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)     # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)     # [batch, 768]
    
    
def simcse_unsup_loss(y_pred: 'tensor') -> 'tensor':
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size * 2, 768]
    
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=DEVICE)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=DEVICE) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return torch.mean(loss)


def eval(model, dataloader) -> float:
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    # label_array = np.array([])
    with torch.no_grad():
        for source, target in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
            # label_array = np.append(label_array, np.array(label))
    # corrcoef 
    return sim_tensor
    # return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
def inference(model, dataloader) -> float:
    """模型评估函数 
    批量预测, batch结果拼接, 一次性求spearman相关度
    """
    model.eval()
    sim_tensor = torch.tensor([], device=DEVICE)
    # label_array = np.array([])
    with torch.no_grad():
        for source, target, _ in dataloader:
            # source        [batch, 1, seq_len] -> [batch, seq_len]
            source_input_ids = source.get('input_ids').squeeze(1).to(DEVICE)
            source_attention_mask = source.get('attention_mask').squeeze(1).to(DEVICE)
            source_token_type_ids = source.get('token_type_ids').squeeze(1).to(DEVICE)
            source_pred = model(source_input_ids, source_attention_mask, source_token_type_ids)
            # target        [batch, 1, seq_len] -> [batch, seq_len]
            target_input_ids = target.get('input_ids').squeeze(1).to(DEVICE)
            target_attention_mask = target.get('attention_mask').squeeze(1).to(DEVICE)
            target_token_type_ids = target.get('token_type_ids').squeeze(1).to(DEVICE)
            target_pred = model(target_input_ids, target_attention_mask, target_token_type_ids)
            # concat
            sim = F.cosine_similarity(source_pred, target_pred, dim=-1)
            sim_tensor = torch.cat((sim_tensor, sim), dim=0)            
            # label_array = np.append(label_array, np.array(label))
    # corrcoef 
    # return spearmanr(label_array, sim_tensor.cpu().numpy()).correlation
    return sim

            
def train(model, train_dl, optimizer) -> None:
    """模型训练函数"""
    model.train()
    global best
    for batch_idx, source in enumerate(tqdm(train_dl), start=1):
        # 维度转换 [batch, 2, seq_len] -> [batch * 2, sql_len]
        real_batch_num = source.get('input_ids').shape[0]
        input_ids = source.get('input_ids').view(real_batch_num * 2, -1).to(DEVICE)
        attention_mask = source.get('attention_mask').view(real_batch_num * 2, -1).to(DEVICE)
        token_type_ids = source.get('token_type_ids').view(real_batch_num * 2, -1).to(DEVICE)
        
        out = model(input_ids, attention_mask, token_type_ids)        
        loss = simcse_unsup_loss(out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 1000 == 0:     
            logger.info(f'loss: {loss.item():.4f}')
            # corrcoef = eval(model, dev_dl)
            # model.train()
            # if best < corrcoef:
                # best = corrcoef
            torch.save(model.state_dict(), SAVE_PATH)
            # logger.info(f"higher corrcoef: {best:.4f} in batch: {batch_idx}, save model")
       
            
if __name__ == '__main__':
    logger.info(f'device: {DEVICE}, pooling: {POOLING}, model path: {model_path}')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_data,dev_data  = load_data(train_file_data_pos,train_file_data_neg)    
    train_dataloader = DataLoader(TrainDataset(train_data), batch_size=BATCH_SIZE,shuffle=True)
    # dev_dataloader = DataLoader(TrainDataset(dev_data), batch_size=BATCH_SIZE,shuffle=False)
    # test_dataloader = DataLoader(TestDataset(test_data), batch_size=BATCH_SIZE)
    # load model    
    assert POOLING in ['cls', 'pooler', 'last-avg', 'first-last-avg']
    model = SimcseModel(pretrained_model=model_path, pooling=POOLING)
    model.to(DEVICE)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    # # train
    # best = 0
    # for epoch in range(EPOCHS):
    #     logger.info(f'epoch: {epoch}')
    #     train(model, train_dataloader, optimizer)
    # logger.info(f'train is finished, best model is saved at {SAVE_PATH}')
    sentence = '放你妈的屁，这个婊子该被人干死'
    id2text,vecs,all_ids = encode_file(test_data,model)
    index = build_index(vecs,all_ids,nlist=1024)
    index.nprob = 20
    sim_sentences = sim_query(index,sentence,model,id2text,topK=10)
    for line in (sim_sentences):
        print(line)
