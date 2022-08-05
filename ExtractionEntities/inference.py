import os
import torch
from transformers import BertTokenizer, BertModel, BertConfig,BertTokenizerFast
from utils.tools import token_rematch,setup_seed
import numpy as np
from model.model import GlobalPointerNet
from data_processing.data_process import yeild_data
import json
from torch import nn
from tqdm import tqdm

import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="biaffine", choices=['biaffine', 'UnlabeledEntity', 'globalpointer'], help='choice model type')
parser.add_argument('--config_file', type=str, default="./train_config/config_yang.ini",  help='choice config file')
args = parser.parse_args()
setup_seed(1234)
from Config import Config
config = Config(args.config_file,args.model_type)
from utils.Logginger import init_logger
logger = init_logger("ner", logging_path=config.log_path)
tokenizer = BertTokenizerFast.from_pretrained(config.model_path,do_lower_case= True)

_,categories_size,_,id2categories = yeild_data(
    config.train_file_data,
    is_train=True,
    maxlen=config.maxlen,
    batch_size=config.batch_size,
    tokenizer=tokenizer,
    DDP=False
)

# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
def get_mapping(text):
    text_token = tokenizer.tokenize(
        text,
        max_length=config.max_length,
        add_special_tokens=True
    )
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
        encode_dict = tokenizer(
            text,
            return_offsets_mapping=True,
            max_length=config.max_length,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encode_dict['input_ids'].to(config.device)
        token_type_ids = encode_dict['token_type_ids'].to(config.device)
        attention_mask = encode_dict['attention_mask'].to(config.device)
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
                        (mapping[start][0], mapping[end][-1], id2categories[l],text[int(mapping[start][0]):int(mapping[end][-1]+1)])
                        #(mapping[start][0], mapping[end][-1], id2categories[l])
                        # (offset_mapping[start][0], offset_mapping[end][-1], id2categories[l])
                    )
        return entities

NER = NamedEntityRecognizer()
def predict_to_file(in_file, out_file,categories_size,id2categories):
    """预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    print("define model!")
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
            config.categories_size,
            config.head_size,
            config.hidden_size
        ).to(config.device)
        # model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])
        model = DDP(model,device_ids=[config.local_rank],find_unused_parameters=True)

    print("loading model!")
    print(f"{config.model_save_path}_{config.model_type}.pth")
    model.load_state_dict(torch.load(f"{config.model_save_path}_{config.model_type}.pth"))
    data = json.load(open(in_file))
    for d in tqdm(data, ncols=100):
        d['entities'] = []
        entities = NER.recognize(d['text'],categories_size,id2categories,model)
        for e in entities:
            d['entities'].append({
                'start_idx': e[0],
                'end_idx': e[1],
                'type': e[2],
                'entity': e[3]
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
    predict_to_file(config.test_file_data,config.out_file,categories_size,id2categories)
