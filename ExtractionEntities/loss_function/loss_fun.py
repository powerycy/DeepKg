import torch
from model import sequence_masking,sequence_masking_mask
import random
from tools import setup_seed
import torch.nn.functional as F
setup_seed(1234)
def select_real_value(value,mask):
    # value = sequence_masking_mask(value,mask,'-inf',value.ndim - 2)
    # value = sequence_masking_mask(value,mask,'-inf',value.ndim - 1)
    value_triu =  torch.triu(value,diagonal=0)
    value_ones = torch.ones_like(value) * 1e12
    value_tril = torch.tril(value_ones,diagonal=-1)
    value_res = value_triu + value_tril
    value_res = torch.reshape(value_res,(1,-1))
    # value_index = torch.nonzero(value)[:,1]
    
    # value = torch.index_select(value,1,value_index)
    value_index_mask = torch.nonzero(value_res != 1e12)[:,1]
    # value = torch.index_select(value,1,value_index_mask)
    return value_index_mask
def find_pre_index(value):
    value =  torch.triu(value,diagonal=0)
    value = torch.reshape(value,(1,-1))
    value_index = torch.nonzero(value == 1e12)[:,1]
    # value = torch.index_select(value,1,value_index)
    return value_index
    

def select_ones(pre_index,value,y_pred):
    neg_rate = 0.3
    value = torch.reshape(value,(1,-1))
    # value_len = value.shape[-1]
    # value = torch.index_select(value,1,pre_index)
    value_index = torch.nonzero(value)[:,1]
    superset = torch.cat([pre_index, value_index])
    uniset, count = superset.unique(return_counts=True)
    mask = (count == 1)
    pre_index = uniset.masked_select(mask)
    neg_num = int(pre_index.shape[-1] * neg_rate)
    value_random_index = torch.multinomial(pre_index.float(),neg_num,replacement=False)
    pre_index = torch.index_select(pre_index,0,value_random_index)
    y_replace_one = torch.ones_like(y_pred)
    y_replace_one_mask = torch.ones_like(y_pred)
    # value_random_index = torch.tensor(random.sample(value_index,neg_num))
    # value_index = torch.index_select(value_index,0,value_random_index)
    y_replace_one_mask.index_fill_(1,pre_index,(1e12 + 1))

    y_replace_mask = y_replace_one_mask - y_replace_one
    y_pred = y_pred  - y_replace_mask
    # padding = (0,value_len - value.shape[-1])
    # value = F.pad(value,padding)
    return y_pred
def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred,mask,is_train):
    """给GlobalPointer设计的交叉熵
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    if is_train:
        y_pred_trans = y_pred.clone()
        y_true_trans = y_true.clone()
        y_pred = torch.reshape(y_pred,(1,-1))
        # y_true_mask = select_ones(y_true_trans)
        # y_pre_index = find_pre_index(y_pre_trans)
        y_pred_trans_index = select_real_value(y_pred_trans,mask)
        # y_pred_trans * mask
        y_pred = select_ones(y_pred_trans_index,y_true_trans,y_pred)
        # y_pred = sequence_masking(y_pred,y_true_mask,'-inf',y_pred.ndim - 2)
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))
