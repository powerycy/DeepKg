import numpy as np
from collections import Counter
from tqdm import tqdm
import pandas as pd
import pickle
from rank_bm25 import BM25Okapi
import random
import sys
import os.path
sys.path.append('..')
from tools.loader import load_json
from tools.text_preprocess import *
from tools.common_tools import timer

class BM25_Model(object):
    def __init__(
            self, is_cut_status="char",
            save_path = "resource/",  
            stop_word_path="stopword.txt"
        ):
        # 分词处理函数 适配器
        self.cut_fun_dic = {
            "cut":cutWordDataProcess,
            "char":cutCharDataProcess,
            "ngram":ngramDataProcess,
        }
        # cut 结巴分词 har 字分 ngram  2-gram 
        self.is_cut_status = is_cut_status if is_cut_status in self.cut_fun_dic else "char" 
        # 获取停用词
        self.stoplist = getStopWord(f"{save_path}{stop_word_path}")
        self.save_path = save_path
        self.fila_name = "bm25"
        if not self.load_plk(self.save_path,self.fila_name,self.is_cut_status):
            print("Need build model before Predict!!!")

    # 功能：构建 模型
    @timer
    def build_model(self, documents_list):
        '''
            功能：构建 模型 
            input： 
                documents_list       List        doc 列表 
           return:
               
        '''
        self.origin_documents_list = documents_list
        self.documents_list = documents_list
        self.bm25 = BM25Okapi(self.documents_list)
        # BM25 模型存储
        self.save_plk(self.save_path,self.fila_name,self.is_cut_status)
    
    # 功能：计算 用户 query 和 标准库 中 候选 query 的 BM25 分数
    def get_documents_score(self, query, topN = 5):
        '''
            功能：计算 用户 query 和 标准库 中 候选 query 的 BM25 分数
            input：
                query         String       用户输入 
                topN          int         选取 TopN
            return:
                candidate_query_score_list    Dict List 与 用户 query 相似的候选query和分数
        '''
        query = self.cut_fun_dic[self.is_cut_status]([query],self.stoplist)[0]
        candidate_query_score_list = []
        doc_scores = self.bm25.get_scores(query)
        
        for q,score in zip(self.origin_documents_list,doc_scores):
            candidate_query_score_list.append({
                "query":q,"score":score
            })
         
        candidate_query_score_list = sorted(candidate_query_score_list, key=lambda x: x['score'], reverse=True)
        return candidate_query_score_list[:topN]

    # 功能：获取 与 用户 query 相似的候选query
    def get_candidate_query(self, query, topN = 5):
        '''
            功能：获取 与 用户 query 相似的候选query
            input：
                query         String       用户输入 
                topN          int         选取 TopN
            return:
                candidate_query_list    Dict List 与 用户 query 相似的候选query
        '''
        candidate_query_list = [candidate_query_score['query'] for candidate_query_score in self.get_documents_score(query, topN = topN)]
        return candidate_query_list

    # 功能：字典数据 存储 为 plk
    def save_plk(self,save_path,fila_name,is_cut_status):
        '''
            功能：字典数据 存储 为 plk
            input: 
                save_path     String    存储目录 
                fila_name     String    存储文件 
                is_cut_status String    
            return:
        '''
        dic = {
           "bm25_model": self.bm25,
            "is_cut_status": self.is_cut_status,
            "stoplist": self.stoplist,
            "origin_documents_list": self.origin_documents_list,
        }
        model_file = save_path +"output/"+ fila_name+"_"+is_cut_status + '.pkl'
        if not os.path.isfile(model_file):
            with open(model_file, 'wb') as f:
                pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL) 
   
    # 功能：加载 plk 中 字典数据
    def load_plk(self,save_path,fila_name,is_cut_status):
        '''
            功能：加载 plk 中 字典数据
            input: 
                save_path     String    存储目录 
                fila_name     String    存储文件 
                is_cut_status String    
            return:
                dic        Dict     字典数据 
        '''
        dic = {}
        model_file = save_path +"output/"+ fila_name+"_"+is_cut_status + '.pkl'
        if os.path.isfile(model_file):
            with open(model_file, 'rb') as f:
                dic = pickle.load(f) 
            self.is_cut_status = dic['is_cut_status']
            self.bm25 = dic['bm25_model']
            self.stoplist = dic['stoplist']
            self.origin_documents_list = dic['origin_documents_list']
            return 1
        else:
            return 0

if __name__ == "__main__":
    # 变量设置
    origin_data_path = "F:/document/datasets/nlpData/CBLUE/CHIP-CDN/"
    file_name = f"{origin_data_path}国际疾病分类 ICD-10北京临床版v601.xlsx"
    data_path = "resource/"
    is_cut_status="cut"
    topN = 100    
    file_name_list = [
        "CHIP-CDN_train","CHIP-CDN_dev"
    ]  
    false_num_rate = 5

    # 训练数据加载
    df = pd.read_excel(file_name,sheet_name='Sheet1',header=None)
    df.columns=['id','name']
    sentences = list(df['name'])

    # B25 模型训练
    bm25_model = BM25_Model(is_cut_status=is_cut_status, save_path = data_path)
    bm25_model.build_model(sentences)
    
    # 验证数据加载
    data_list = load_json(f"{origin_data_path}{file_name_list[1]}.json")
    query_list = bm25_model.get_candidate_query(data_list[0]['text'],topN=topN)
    '''
        print(f"query_list:{query_list}")
        >>>
        query_list:['1型糖尿病性单神经病', '1型糖尿病性股神经病', '1型糖尿病性神经根病', '2型糖尿病性单神经病', '2型糖尿病性股神经病', '2型糖尿病性神经根病', '1型糖尿病性胸神经根病', '1型糖尿病性周围神经病', '1型糖尿病性自主神经病', '2型糖尿病性胸神经根病']
    '''

    documents_score_list = bm25_model.get_documents_score(data_list[0]['text'],topN=topN)
    '''
        print(f"documents_score_list:{documents_score_list}")
        >>>
        documents_score_list:[{'query': '1型糖尿病性单神经病', 'score': 23.498723478171133}, {'query': '1型糖尿病性股神经病', 'score': 23.498723478171133}, {'query': '1型糖尿病性神经根病', 'score': 23.498723478171133}, {'query': '2型糖尿病性单神经 病', 'score': 23.498723478171133}, {'query': '2型糖尿病性股神经病', 'score': 23.498723478171133}, {'query': '2型糖尿病性神经根病', 'score': 23.498723478171133}, {'query': '1型糖尿病性胸神经根病', 'score': 22.558958194390154}, {'query': '1型糖尿病性周围神经病', 'score': 22.558958194390154}, {'query': '1型糖尿病性自主神经病', 'score': 22.558958194390154}, {'query': '2型糖尿病性胸神经根病', 'score': 22.558958194390154}]
    '''



