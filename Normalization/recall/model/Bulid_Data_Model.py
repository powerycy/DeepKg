from tqdm import tqdm
import sys
import os.path
import pandas as pd
sys.path.append('..')
from model.BM25_Model import BM25_Model
from model.ES_Model import ES_Model
from tools.loader import load_json,write_json
from tools.multiprocessing_tools import Multiprocessing_class

class Bulid_Data_Model(object):
    def __init__(
            self, model_type, save_path = "resource/",
            is_cut_status="cut", indices = "cdn", doc_type = "cdn", host="127.0.0.1"
        ):
        self.save_path = save_path
        if model_type == "bm25":
            # B25 模型训练
            self.model = BM25_Model(is_cut_status=is_cut_status, save_path = save_path)
        elif model_type == "es":
            # B25 模型定义
            self.model = ES_Model(indices = indices, doc_type = doc_type, host=host)
            
    # 功能：将 国际疾病分类 ICD-10北京临床 导入 es or BM2.5
    def build_model(self,file_name):
        '''
            功能：将 国际疾病分类 ICD-10北京临床 导入 es or BM2.5
            input:
                file_name       String      国际疾病分类 ICD-10北京临床 文件路径
            return：
        '''
        df = pd.read_excel(file_name,sheet_name='Sheet1',header=None)
        df.columns=['id','name']
        sentences = list(df['name'])
        self.model.build_model(sentences)

    # 功能：对 训练集 构建候选 query
    def build_train_candidate_query(self, data_list, topN, false_num_rate=1):
        '''
            功能：对 测试集 构建候选 query
            input:
                data_list        Dict List     数据
                topN             int           选取数量 
                false_num_rate        int            采样比例 
            return:
                result_dict_list Dict List     包含 候选query 的数据
        '''
        result_dict_list = self.select_candidate_query(data_list, topN)
        simple_data_list = self.build_sample(result_dict_list, false_num_rate)
        return simple_data_list

    # 功能：对 测试集 构建候选 query
    def build_test_candidate_query(self, data_list, topN):
        '''
            功能：对 测试集 构建候选 query
            input:
                data_list        Dict List     数据
                topN             int           选取数量 
            return:
                result_dict_list Dict List     包含 候选query 的数据
        '''
        result_dict_list = []
        for data in tqdm(data_list):
            result_dict_list.append({
                "query" : data['text'],
                "candidate_query": self.model.get_documents_score(data['text'],topN=topN)
            })
        return result_dict_list
    
    # 功能：召回的候选 query 做分析
    def candidate_query_analysis(self, data_list, topN=100):
        '''
            功能：召回的候选 query 做分析
            input:
                data_list     List(String)     query 列表
                topN             int           选取数量   
            return:
                Score         List(int)        相似度分数  
        '''
        sim_score = 0
        for data in tqdm(data_list):
            candidate_query_set = set(self.model.get_candidate_query(data['text'],topN=topN))
            normalized_result_list = data['normalized_result'].split("##")
            score = 0
            for normalized_result in normalized_result_list:
                if normalized_result in candidate_query_set:
                    score = score+1
            sim_score += score/len(normalized_result_list)
        return [sim_score/len(data_list)]
    
    # 功能：获取 候选 query
    def select_candidate_query(self, data_list, topN):
        '''
            功能：获取 候选 query
            input:
                data_list        Dict List     数据
                topN             int           选取数量 
            return:
                result_dict_list Dict List     包含 候选query 的数据
        '''
        result_dict_list = []
        for data in tqdm(data_list):
            result_dict_list.append({
                "query" : data['text'],
                "normalized_result":data['normalized_result'].split("##"),
                "candidate_query": self.model.get_documents_score(data['text'],topN=topN)
            })
        return result_dict_list

    # 功能：构建 训练 测试集
    def build_sample(self, result_dict_list, false_num_rate=1):
        '''
            功能：构建 训练 测试集
            input:
                result_dict_list       Dict List       ES 检索集
                false_num_rate        int            采样比例 
            return:
                sample_list          Dict List        训练 测试集
        '''
        sample_list = []
        for index,result_dict in tqdm(enumerate(result_dict_list)):
            # 构建正样本
            for query in result_dict['normalized_result']:
                sample_list.append({
                    "text1" : result_dict['query'],
                    "text2" : query,
                    "label" : 1
                })
            
            false_num = 0
            i = 0
            has_query_set = set(result_dict['normalized_result'])
            while false_num<len(result_dict['normalized_result'])*false_num_rate and i < len(result_dict['candidate_query']):
                if result_dict['candidate_query'][i]['query'] not in has_query_set:
                    sample_list.append({
                        "text1" : result_dict['query'],
                        "text2" : result_dict['candidate_query'][i]['query'],
                        "label" : 0
                    })
                    has_query_set.add(result_dict['candidate_query'][i]['query'])
                    false_num= false_num+1
                i = i+1
        return sample_list  

