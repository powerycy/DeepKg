import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel, BertConfig,PreTrainedTokenizerFast,BertTokenizerFast
import json
from utils.tools import search
from utils.tools import token_rematch
from torch.utils.data.distributed import DistributedSampler

def load_data(filename,is_train):
    
    resultList = []
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename,encoding="utf-8")):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            resultList.append(label)
    categories = list(set(resultList))
    categories.sort(key=resultList.index)
    if is_train:
        return D,categories
    else:
        return D
class NerDataset(Dataset):
    def __init__(self, data, tokenizer,categories_size,categories2id, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.categories_size = categories_size
        self.categories2id = categories2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((self.categories_size,self.maxlen,self.maxlen))
        context = self.tokenizer(d[0],return_offsets_mapping=True,max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        tokens = self.tokenizer.tokenize(d[0],max_length=self.maxlen,add_special_tokens=True)
        mapping = token_rematch().rematch(d[0],tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        for entity_input in d[1:]:
            start, end = entity_input[0], entity_input[1]
            if start in start_mapping and end in end_mapping and start < self.maxlen and end < self.maxlen:
                start = start_mapping[start]
                end = end_mapping[end]
                label[self.categories2id[entity_input[2]],start,end] = 1
        return context,label

def yeild_data(train_file_data, is_train, maxlen, batch_size, tokenizer, categories_size=None,categories2id=None,DDP=True):
    if is_train:
        train_data, categories = load_data(train_file_data,is_train=is_train)
        categories_size = len(categories)
        categories2id = {c:idx for idx,c in enumerate(categories)}
        id2categories = {idx : c for idx,c in enumerate(categories)}
        train_data = NerDataset(train_data,tokenizer,categories_size,categories2id, maxlen)
        if DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, batch_size=batch_size,sampler=train_sampler,shuffle=False)
        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
        return train_dataloader,categories_size,categories2id,id2categories
    else:
        train_data = load_data(train_file_data,is_train=is_train)
        train_data = NerDataset(train_data,tokenizer,categories_size,categories2id, maxlen)
        if DDP:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, batch_size=batch_size,sampler=train_sampler)
        else:
            train_dataloader = DataLoader(train_data, batch_size=batch_size)
        return train_dataloader

# if __name__ == '__main__':
#     # train_data, categories = load_data('/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_train.json')
#     # val_data = load_data('/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_dev.json',is_train=False)
#     # categories_size = len(categories)
#     # categories2id = {c:idx for idx,c in enumerate(categories)}
#     # id2categories = {idx : c for idx,c in enumerate(categories)}
    # train_dataloader,val_dataloader,categories_size = yeild_data('/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_train.json','/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_dev.json')
#     a = 1
