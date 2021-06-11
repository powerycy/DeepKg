import configparser
import torch
con = configparser.ConfigParser()
class Config():
    def __init__(self,config_file,model_type):
        file = config_file
        self.model_type = model_type
        con.read(file,encoding='utf8')
        items = con.items('path')
        path = dict(items)
        items = con.items('model_es')
        model_es = dict(items)
        items = con.items('model_bm25')
        model_bm25 = dict(items)
        self.origin_data_path = path['origin_data_path']
        self.file_name = f"{path['origin_data_path']}{path['file_name']}"
        self.data_path = path['data_path']
        self.topN = eval(path['topn'])
        self.file_name_list = eval(path['file_name_list'])
        self.test_file_name = path['test_file_name']
        self.processes_num = eval(path['processes_num'])
        self.is_build_model = eval(path['is_build_model'])
        self.false_num_rate = eval(path['false_num_rate'])
        self.is_cut_status = model_bm25['is_cut_status']

config_file = "config.ini"
model_type = "bm25"
config = Config(config_file,model_type)