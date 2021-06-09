import unicodedata, re
import torch
import torch.distributed as dist
def search(offset_mapping, index):
    for idx, internal in enumerate(offset_mapping[1:]):
        if internal[0] <= index < internal[1]:
            return idx
    return None
def reduce_tensor(tensor: torch.Tensor,reduce_loss) -> torch.Tensor:
    rt = tensor.detach()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    if reduce_loss:
        rt /= dist.get_world_size()#进程数
        return rt
    else:
        return rt 
class token_rematch:
    def __init__(self):
        self._do_lower_case = True


    @staticmethod
    def stem(token):
            """获取token的“词干”（如果是##开头，则自动去掉##）
            """
            if token[:2] == '##':
                return token[2:]
            else:
                return token
    @staticmethod
    def _is_control(ch):
            """控制类字符判断
            """
            return unicodedata.category(ch) in ('Cc', 'Cf')
    @staticmethod
    def _is_special(ch):
            """判断是不是有特殊含义的符号
            """
            return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([])
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end
        return token_mapping

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # self.shadow[name] = param.data.clone()
                self.shadow[name] = param.data.detach()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                # self.shadow[name] = new_average.clone()
                self.shadow[name] = new_average.detach()


    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
