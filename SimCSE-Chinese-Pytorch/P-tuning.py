from transformers import BigBirdTokenizer, BigBirdForMaskedLM
import torch
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')

PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)   # PAD_INDEX = 0
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)   # UNK_INDEX = 100
MASK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) # MASK_INDEX = 67
""" 因为 BigBird 的 tokenizer 里面没有 "opinion" 和 "unused1"... 的 token，所以我们要先自己新加上这几个 token. """
tokenizer.add_tokens('opinion')
desc = ['[unused%s]' % i for i in range(1, 9)]
for item in desc:
    tokenizer.add_tokens(item)
import numpy as np

def random_masking(token_ids):
    """ 对输入进行随机 mask """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(MASK_INDEX)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer.vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target
mask_idx = 5
desc = ['[unused%s]' % i for i in range(1, 9)]
desc.insert(mask_idx - 1, tokenizer.mask_token)
desc_ids = [tokenizer.convert_tokens_to_ids(t) for t in desc]

pos_id = tokenizer.convert_tokens_to_ids('opinion')
neg_id = tokenizer.convert_tokens_to_ids('report')
from torch.utils.data import Dataset

class MaskDataset(Dataset):
    def __init__(self, tokenizer, data, max_len, random=True):

        self.token_ids_list, self.output_ids_list = [], []
        for item in data:
            data_text = item[0]
            data_label = item[1]

            token_ids = tokenizer.encode(data_text)
            token_ids = token_ids[:max_len]

            if data_label != 2:
                token_ids = token_ids[:1] + desc_ids + token_ids[1:]
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]

            if data_label == 0:
                source_ids[mask_idx] = MASK_INDEX
                target_ids[mask_idx] = neg_id
            elif data_label == 1:
                source_ids[mask_idx] = MASK_INDEX
                target_ids[mask_idx] = pos_id

            source_ids.extend((max_len+9-len(source_ids))*[PAD_INDEX])
            target_ids.extend((max_len+9-len(target_ids))*[PAD_INDEX])

            self.token_ids_list.append(source_ids)
            self.output_ids_list.append(target_ids)

        if len(self.token_ids_list) != len(self.output_ids_list):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.token_ids_list)

    def __getitem__(self, index):
        # note that this isn't randomly selecting. It's a simple get a single item that represents an x and y
        _x = self.token_ids_list[index]
        _y = self.output_ids_list[index]

        return _x, _y
def collate_fn(data):
    unit_x, unit_y = [], []
    for item in data:
        unit_x.append(item[0])
        unit_y.append(item[1])
    return {torch.tensor(unit_x), torch.tensor(unit_y)}
from transformers import BigBirdForMaskedLM

""" model.resize_token_embeddings(len(tokenizer))  要加上这句话，因为新加了 token """
model = BigBirdForMaskedLM.from_pretrained('google/bigbird-roberta-base')
model.resize_token_embeddings(len(tokenizer))  
model.to(device)