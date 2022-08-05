import os
import json
# 功能：加载 json 文件 
def load_json(json_file):
    '''
        功能：加载 json 文件 
        input：
            json_file   String    配置文件 
        return:
            load_dict   Dict     配置项
    '''
    with open(json_file,'r',encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict

# 功能：存储 json 
def write_json(file,dic):
    '''
        功能：存储 json 
        input：
            file   String    存储文件 
            dic    Dic      字典文件
        return:
        
        函数参数介绍：
            json.dumps():
                ensure_ascii      bool         存储编码格式，False 表示 中文 
                indent          int          缩进  
    '''
    with open(file, 'w', encoding="utf-8") as f:
        f.write(json.dumps(dic,ensure_ascii=False, indent=2))

import pickle
# 功能：字典数据 存储 为 plk
def save_plk_dict(dic,save_path,fila_name):
    '''
        功能：字典数据 存储 为 plk
        input:
        dic          Dict     存储字典    
        save_path     String    存储目录 
        fila_name     String    存储文件 
    return:

    '''
    with open(save_path+ fila_name + '.pkl', 'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL) 
