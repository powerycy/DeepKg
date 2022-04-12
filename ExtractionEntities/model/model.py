import torch
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import nn
from transformers import BertModel
import math
from torch.nn import functional as F
# import argparse
# parser = argparse.ArgumentParser()
# args = parser.parse_args()
class SinusoidalPositionEmbedding(Module):
    """定义Sin-Cos位置Embedding
    """
    def __init__(
        self, output_dim, merge_mode='add', custom_position_ids=False):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def forward(self, inputs):
        if self.custom_position_ids:
            seq_len = inputs.shape[1]
            inputs,position_ids = inputs
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(seq_len).type(torch.float)[None]
        indices = torch.arange(self.output_dim // 2).type(torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, seq_len, self.output_dim))
        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)
#相对位置编码
def relative_position_encoding(depth, max_length=512, max_relative_position=127):
    vocab_size = max_relative_position * 2 + 1
    range_vec = torch.arange(max_length)
    range_mat = range_vec.repeat(max_length).view(max_length, max_length)
    distance_mat = range_mat - torch.t(range_mat)
    distance_mat_clipped = torch.clamp(distance_mat, -max_relative_position, max_relative_position)
    final_mat = distance_mat_clipped + max_relative_position

    embeddings_table = torch.zeros(vocab_size, depth)
    position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
    embeddings_table[:, 0::2] = torch.sin(position * div_term)
    embeddings_table[:, 1::2] = torch.cos(position * div_term)
    embeddings_table = embeddings_table.unsqueeze(0).transpose(0, 1).squeeze(1)

    flat_relative_positions_matrix = final_mat.view(-1)
    one_hot_relative_positions_matrix = torch.nn.functional.one_hot(flat_relative_positions_matrix,
                                                                    num_classes=vocab_size).float()
    positions_encoding = torch.matmul(one_hot_relative_positions_matrix, embeddings_table)
    my_shape = list(final_mat.size())
    my_shape.append(depth)
    positions_encoding = positions_encoding.view(my_shape)
    return positions_encoding
def sequence_masking(x, mask, value='-inf', axis=None):
    if mask is None:
        return x
    else:
        if value == '-inf':
            value = -1e12
        elif value == 'inf':
            value = 1e12
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = torch.unsqueeze(mask, 1)
        for _ in range(x.ndim - mask.ndim):
            mask = torch.unsqueeze(mask, mask.ndim)
        return x * mask + value * (1 - mask)
def add_mask_tril(logits, mask):
    if mask.dtype != logits.dtype:
        mask = mask.type(logits.dtype)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 2)
    logits = sequence_masking(logits, mask, '-inf', logits.ndim - 1)
    # 排除下三角
    mask = torch.tril(torch.ones_like(logits), diagonal=-1)
    logits = logits - mask * 1e12
    return logits

class Biaffine(Module):
    def __init__(self, in_size, out_size,dim_in, dim_hid,abPosition = False,bias_x = True,bias_y = True):
        super(Biaffine, self).__init__()
        self.out_size = out_size
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight1 = Parameter(torch.Tensor(in_size+1, out_size, in_size + 1))
        self.weight2 = Parameter(torch.Tensor(2 * in_size + 3, out_size))
        self.start_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*dim_in, out_features=dim_hid),
                                            torch.nn.ReLU())
        self.end_layer = torch.nn.Sequential(torch.nn.Linear(in_features=2*dim_in, out_features=dim_hid),
                                            torch.nn.ReLU())
        self.lstm=torch.nn.LSTM(input_size=dim_in,hidden_size=dim_in, \
                    num_layers=1,batch_first=True, \
                    dropout=0.3,bidirectional=True)
        self.abPosition = abPosition
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight1,a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight2,a=math.sqrt(5))
    
    def forward(self, inputs,mask = None):
        input_length = inputs.shape[1]
        hidden_size = inputs.shape[-1]
        if self.abPosition:
            #绝对位置编码
            inputs = SinusoidalPositionEmbedding(hidden_size, 'add')(inputs)
        encoder_rep,_ = self.lstm(inputs)
        start_logits = self.start_layer(encoder_rep) 
        end_logits = self.end_layer(encoder_rep)
        if self.bias_x:
            start_logits = torch.cat((start_logits, torch.ones_like(start_logits[..., :1])), dim=-1)
        if self.bias_y:
            end_logits = torch.cat((end_logits, torch.ones_like(end_logits[..., :1])), dim=-1)
        start_logits_con = torch.unsqueeze(start_logits, 1)
        end_logits_con = torch.unsqueeze(end_logits, 2)
        start_logits_con = start_logits_con.repeat(1, input_length, 1, 1)
        end_logits_con = end_logits_con.repeat(1, 1, input_length, 1)
        concat_start_end = torch.cat([start_logits_con, end_logits_con], dim=-1)
        concat_start_end = torch.cat([concat_start_end, torch.ones_like(concat_start_end[..., :1])],dim=-1)
        # bxi,oij,byj->boxy
        logits_1 = torch.einsum('bxi,ioj,byj -> bxyo', start_logits, self.weight1, end_logits)
        logits_2 = torch.einsum('bijy,yo -> bijo', concat_start_end, self.weight2)
        logits = logits_1 + logits_2
        logits = logits.permute(0,3,1,2)
        logits = add_mask_tril(logits, mask)
        return logits
class BiaffineNet(nn.Module):
    def __init__(self,model_path,categories_size,hidden_size,dim_in,dim_hid,abPosition):
        super(BiaffineNet, self).__init__()
        self.categories_size = categories_size
        self.biaffine = Biaffine(hidden_size, self.categories_size,dim_in = dim_in,dim_hid = dim_hid,abPosition = abPosition)
        self.bert = BertModel.from_pretrained(model_path)
    def forward(self, input_ids, attention_mask, token_type_ids,is_train = True):
        bert_encoder = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_encoder = bert_encoder['last_hidden_state']
        logits = self.biaffine(bert_encoder, mask = attention_mask)
        return logits
class GlobalPointer(Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    """
    def __init__(self, heads, head_size,hidden_size,RoPE=True):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.linear = nn.Linear(hidden_size,self.head_size * self.heads * 2)

    def forward(self, inputs, mask=None):
        inputs = self.linear(inputs)
        inputs = torch.split(inputs, self.head_size * 2 , dim=-1)
        #对输入张量序列进行连接
        inputs = torch.stack(inputs, dim=-2)
        #分出qw和kw
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            cos_pos = pos[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos[..., ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], 3)
            qw2 = torch.reshape(qw2, qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], 3)
            kw2 = torch.reshape(kw2, kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd , bnhd -> bhmn', qw, kw)
        # 排除padding跟下三角
        logits = add_mask_tril(logits,mask)
        return logits / self.head_size ** 0.5
class GlobalPointerNet(nn.Module):
    def __init__(self,model_path,categories_size,head_size,hidden_size):
        super(GlobalPointerNet, self).__init__()
        self.categories_size = categories_size
        self.globalpointer = GlobalPointer(self.categories_size, head_size, hidden_size)
        self.bert = BertModel.from_pretrained(model_path)
    def forward(self, input_ids, attention_mask, token_type_ids,is_train = True):
        bert_encoder = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_encoder = bert_encoder.last_hidden_state
        logits = self.globalpointer(bert_encoder, mask = attention_mask)
        return logits

class UnlabeledEntity(Module):
    def __init__(self,hidden_size,c_size,abPosition = False,rePosition=False, re_maxlen=None,max_relative=None):
        super(UnlabeledEntity, self).__init__()
        self.hidden_size = hidden_size
        self.c_size = c_size
        self.abPosition = abPosition
        self.rePosition = rePosition
        self.Wh = nn.Linear(hidden_size * 4, self.hidden_size)
        self.Wo = nn.Linear(self.hidden_size,self.c_size)
        if self.rePosition:
            self.relative_positions_encoding = relative_position_encoding(max_length=re_maxlen,
                                                                     depth=4*hidden_size,
                                                                     max_relative_position=max_relative)
    def forward(self, inputs, mask=None):
        input_length = inputs.shape[1]
        batch_size = inputs.shape[0]
        if self.abPosition:
            # #绝对位置编码
            inputs = SinusoidalPositionEmbedding(self.hidden_size, 'add')(inputs)
        inputs_1 = torch.unsqueeze(inputs, 1)
        inputs_2 = torch.unsqueeze(inputs, 2)
        inputs_1 = inputs_1.repeat(1, input_length, 1, 1)
        inputs_2 = inputs_2.repeat(1, 1, input_length, 1)
        concat_inputs = torch.cat([inputs_2, inputs_1,inputs_2-inputs_1,inputs_2.mul(inputs_1)], dim=-1)
        if self.rePosition:
            relations_keys = self.relative_positions_encoding[:input_length, :input_length, :].to(inputs.device)
            concat_inputs += relations_keys
        hij = torch.tanh(self.Wh(concat_inputs))
        logits = self.Wo(hij)
        logits = logits.permute(0,3,1,2)
        logits = add_mask_tril(logits, mask)
        return logits
class UnlabeledEntityNet(nn.Module):
    def __init__(self,model_path,categories_size,hidden_size,abPosition,rePosition,re_maxlen,max_relative):
        super(UnlabeledEntityNet, self).__init__()
        self.categories_size = categories_size
        self.unlabeledentity = UnlabeledEntity(hidden_size, self.categories_size,abPosition=abPosition,rePosition=rePosition,re_maxlen = re_maxlen,max_relative=max_relative)
        self.bert = BertModel.from_pretrained(model_path)
    def forward(self, input_ids, attention_mask, token_type_ids,is_train = True):
        bert_encoder = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        bert_encoder = bert_encoder.last_hidden_state
        logits = self.unlabeledentity(bert_encoder, mask = attention_mask)
        return logits

class LayerNormImpl(nn.Module):
    __constants__ = ['weight', 'bias', 'eps']
    def __init__(self, args, hidden, eps=1e-5, elementwise_affine=True):
        super(LayerNormImpl, self).__init__()
        self.mode = args.lnv
        self.sigma = args.sigma
        self.hidden = hidden
        self.adanorm_scale = args.adanorm_scale
        self.nowb_scale = args.nowb_scale
        self.mean_detach = args.mean_detach
        self.std_detach = args.std_detach
        if self.mode == 'no_norm':
            elementwise_affine = False
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(hidden))
            self.bias = nn.Parameter(torch.Tensor(hidden))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.mode == 'no_norm':
            return input
        elif self.mode == 'topk':
            T, B, C = input.size()
            input = input.reshape(T*B, C)
            k = max(int(self.hidden * self.sigma), 1)
            input = input.view(1, -1, self.hidden)
            topk_value, topk_index = input.topk(k, dim=-1)
            topk_min_value, top_min_index = input.topk(k, dim=-1, largest=False)
            top_value = topk_value[:, :, -1:]
            top_min_value = topk_min_value[:, :, -1:]
            d0 = torch.arange(top_value.shape[0], dtype=torch.int64)[:, None, None]
            d1 = torch.arange(top_value.shape[1], dtype=torch.int64)[None, :, None]
            input[d0, d1, topk_index] = top_value
            input[d0, d1, top_min_index] = top_min_value
            input = input.reshape(T, B, self.hidden) 
            return F.layer_norm(
                input, torch.Size([self.hidden]), self.weight, self.bias, self.eps)
        elif self.mode == 'adanorm':
            mean = input.mean(-1, keepdim=True)
            std = input.std(-1, keepdim=True)
            input = input - mean
            mean = input.mean(-1, keepdim=True)
            graNorm = (1 / 10 * (input - mean) / (std + self.eps)).detach()
            input_norm = (input - input * graNorm) / (std + self.eps)
            return input_norm*self.adanorm_scale
        elif self.mode == 'nowb':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            if self.mean_detach:
                mean = mean.detach()
            if self.std_detach:
                std = std.detach()
            input_norm = (input - mean) / (std + self.eps)
            return input_norm
        elif self.mode == 'gradnorm':
            mean = input.mean(dim=-1, keepdim=True)
            std = input.std(dim=-1, keepdim=True)
            input_norm = (input - mean) / (std + self.eps)
            output = input.detach() + input_norm
            return output
def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False,lnv=None):
    if lnv is not None:
        if lnv != 'origin':
            return LayerNormImpl(args, normalized_shape, eps, elementwise_affine)
    if not export and torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
