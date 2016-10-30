import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'
gpus = [3,5,6]
import torch
from transformers import BertTokenizer, BertModel, BertConfig,BertTokenizerFast
from utils.tools import token_rematch
import numpy as np
from model.model import GlobalPointerNet
from data_processing.data_process import yeild_data
import json
from torch import nn
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
from tqdm import tqdm
import configparser
con = configparser.ConfigParser()
file = './train_config/config.ini'
con.read(file,encoding='utf8')
items = con.items('path')
path = dict(items)
items = con.items('model_superparameter')
model_sp = dict(items)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_path = path['model_path']
model_save_path = path['model_save_path']
train_file_data = path['train_file_data']
test_file_data = path['test_file_data']
out_file = path['out_file']
max_length = eval(model_sp['inference_maxlen'])
head_size = eval(model_sp['head_size'])
hidden_size = eval(model_sp['hidden_size'])
tokenizer = BertTokenizerFast.from_pretrained(model_path)
_,categories_size,_,id2categories = yeild_data(train_file_data,is_train=True,DDP=False)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def get_mapping(text):
    text_token = tokenizer.tokenize(text,max_length=max_length,add_special_tokens=True)
    # text_token =  text_token
    text_mapping = token_rematch().rematch(text,text_token)
    return text_mapping
class NamedEntityRecognizer(object):
    """命名实体识别器
    """
    def recognize(self, text,categories_size,id2categories,model,threshold=0):
       
        mapping = get_mapping(text)
        # text_res = ''.join(text_res)
        # mapping = torch.tensor([mapping])
        encode_dict = tokenizer(text,return_offsets_mapping=True,max_length=max_length,truncation=True,return_tensors='pt')
        input_ids = encode_dict['input_ids'].to(device)
        token_type_ids = encode_dict['token_type_ids'].to(device)
        attention_mask = encode_dict['attention_mask'].to(device)
        # offset_mapping = encode_dict['offset_mapping'].to(device)
        # token_ids, segment_ids = to_array([token_ids], [segment_ids])
        # scores = model.predict([token_ids, segment_ids])[0]
        scores = model(input_ids,attention_mask,token_type_ids)[0]
        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf
        entities = []
        # threshold = torch.tensor(threshold).to(device)
        scores = scores.detach().cpu().numpy()
        for l, start, end in zip(*np.where(scores > threshold)):
            if start < len(mapping) and end < len(mapping):
                if (len(mapping[start]) and len(mapping[end])) > 0:
                    entities.append(
                        # (mapping[start][0], mapping[end][-1], id2categories[l],text[mapping[start][0]:mapping[end][-1]]+1)
                        (mapping[start][0], mapping[end][-1], id2categories[l])
                        # (offset_mapping[start][0], offset_mapping[end][-1], id2categories[l])
                    )
        return entities
NER = NamedEntityRecognizer()
def predict_to_file(in_file, out_file,categories_size,id2categories):
    """预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    model = GlobalPointerNet(model_path,categories_size,head_size,hidden_size)
    model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
    model.module.load_state_dict(torch.load(model_save_path))
    data = json.load(open(in_file))
    for d in tqdm(data, ncols=100):
        d['entities'] = []
        entities = NER.recognize(d['text'],categories_size,id2categories,model)
        for e in entities:
            d['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2],
                # 'entity': e[3]
            })
    json.dump(
        data,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )
if __name__ == "__main__":
    # in_file = '/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_test.json'
    # val_file_data = '/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_dev.json'
    # train_file_data = '/home/yuanchaoyi/DeepKg/PyTorch_BERT_Biaffine_NER/data/tianchi_data/CBLUE/CMeEE/CMeEE_train.json'
    # out_file = 'result.json'
    # _,categories_size,_,id2categories = yeild_data(train_file_data,is_train=True,DDP=False)
    predict_to_file(test_file_data,out_file,categories_size,id2categories)
