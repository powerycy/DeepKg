from tqdm import tqdm
import sys
import os.path
import pandas as pd
from tools.loader import load_json,write_json
from tools.multiprocessing_tools import Multiprocessing_class
from model.Bulid_Data_Model import Bulid_Data_Model
import argparse
import sys
from Config import Config
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="bm25", choices=['es', 'bm25'], help='choice model type')
parser.add_argument('--config_file', type=str, default="./resource/config/config.ini",  help='choice config file')
args = parser.parse_args()

if __name__ == "__main__":

    config = Config(config_file=args.config_file,model_type=args.model_type)
    
    multiprocessingClass = Multiprocessing_class(config.processes_num)
        
    # 构建数据类 定义
    bulid_data_model = Bulid_Data_Model(config.model_type, save_path = config.data_path)
    if config.is_build_model:
        bulid_data_model.build_model(config.file_name)

    # 文本匹配 训练数据 构建
    for i in range(len(config.file_name_list)):
        data_list = load_json(f"{config.origin_data_path}{config.file_name_list[i]}.json")

        simple_data_list = multiprocessingClass.use_multiprocessing_for_list(
            bulid_data_model.build_train_candidate_query,
            data_list, int(len(data_list)/config.processes_num), config.topN, config.false_num_rate
        )

        pd.DataFrame(simple_data_list).sample(frac=1)[['text1','text2','label']].to_csv(
            f"{config.data_path}output/{config.file_name_list[i]}.txt",
            encoding="utf8", index=None, header=None, sep="\t"
        )
    
    # 分析 模型 召回 效果
    data_list = load_json(f"{config.origin_data_path}{config.file_name_list[1]}.json")
    recall_score = multiprocessingClass.use_multiprocessing_for_list(
        bulid_data_model.candidate_query_analysis,
        data_list, int(len(data_list)/config.processes_num), config.topN
    )
    print(f"recall_score:{sum(recall_score)/len(recall_score)}")

    # 对 测试集 构建候选 query
    data_list = load_json(f"{config.origin_data_path}{config.test_file_name}.json")
    result_dict_list = multiprocessingClass.use_multiprocessing_for_list(
        bulid_data_model.build_test_candidate_query,
        data_list, int(len(data_list)/config.processes_num), config.topN
    )
    write_json(f"{config.data_path}output/{config.test_file_name}_origin_{config.model_type}.json",result_dict_list)



