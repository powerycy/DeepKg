import configparser
import torch
con = configparser.ConfigParser()
class Config():
    def __init__(self,config_file,model_type):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        file = config_file
        self.model_type = model_type
        con.read(file,encoding='utf8')
        items = con.items('path')
        path = dict(items)
        items = con.items('model_superparameter')
        model_sp = dict(items)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.gpus = [0]
            torch.cuda.set_device('cuda:{}'.format(self.gpus[0]))
        self.model_path = path['model_path']
        self.data_path = path['data_path']
        self.log_path = path['log_path']
        self.train_file_data = f"{self.data_path}{path['train_file_data']}"
        self.val_file_data = f"{self.data_path}{path['val_file_data']}"
        self.test_file_data = f"{self.data_path}{path['test_file_data']}"
        self.model_save_path = path['model_save_path']
        self.no_decay = ['bias', 'gamma', 'beta']
        self.epochs = eval(model_sp['epochs'])
        self.clip_norm = eval(model_sp['clip_norm'])
        self.warmup_steps = eval(model_sp['warmup_steps'])
        self.learning_rate = eval(model_sp['learning_rate'])
        self.hidden_size = eval(model_sp['hidden_size'])
        self.head_size = eval(model_sp['head_size'])
        self.abPosition = eval(model_sp['abposition'])
        self.maxlen = eval(model_sp['maxlen'])
        self.batch_size = eval(model_sp['batch_size'])
        self.max_length = eval(model_sp['inference_maxlen'])
        self.out_file = path['out_file']
        if self.model_type=="biaffine":
            self.dim_in = eval(model_sp['dim_in'])
            self.dim_hid = eval(model_sp['dim_hid'])
        elif self.model_type=="UnlabeledEntity":
            self.rePosition= eval(model_sp['reposition'])
            self.max_relative = eval(model_sp['max_relative'])
            self.re_maxlen = eval(model_sp['re_maxlen'])
        elif self.model_type=="globalpointer":
            self.local_rank = eval(model_sp['local_rank'])
            self.re_maxlen = eval(model_sp['re_maxlen'])


        