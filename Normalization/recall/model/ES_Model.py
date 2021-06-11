# encoding=utf8
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import pandas as pd
import sys
sys.path.append('..')
from tools.loader import load_json
from tools.common_tools import timer

class ES_Model(object):
    def __init__(self, indices = "cdn", doc_type = "cdn", host="127.0.0.1"):
        # 分词处理函数 适配器
        self.indices = indices
        self.doc_type = doc_type
        self.mapping = {
                "properties": {
                    "name": {
                        "type": "text",
                        "analyzer": "ik_max_word",
                        "search_analyzer": "ik_max_word"
                    }
                }
            }
        self.es = Elasticsearch(host=host)
    
    # 功能：创建索引
    def create_indices(self):
        '''
            功能：创建索引
            input:

            return:

        '''
        self.es.indices.create(index=self.indices, ignore=400)

    # 功能：构建 模型
    @timer
    def build_model(self, documents_list):
        '''
            功能：使用生成器批量写入数据 
                input:
                es               Elasticsearch 对象 
                query2temp_list      Dict List           待插入数据 
            return  
                status          0/1                失败/成功     
        '''
        # 创建索引
        self.create_indices()

        result = self.es.indices.put_mapping(
            index=self.indices, 
            doc_type=self.doc_type, 
            body=self.mapping
        )
        print(f"len(documents_list):{len(documents_list)}")

        # 插入数据
        action = ({
            "_index": self.indices,
            "_type": self.doc_type,
            "_source": {
                'name': documents
            }
        } for documents in documents_list)
        # try:
        helpers.bulk(self.es, action) 
        # except Exception:
        #     print("数据批量导入出错")
        #     return 0
        # else:
        #     return 1

    # 功能：查询 es ,并筛选候选 query
    @timer
    def select_candidate_query_score(self, data_list,repeat_score = 2,es_size=100,top = 20):
        all_hit_score = 0
        result_dict_list = []
        for data in tqdm(data_list):
            hit = 0                 # 命中数
            normalized_result_set = set(data['normalized_result'].split("##"))
            # es 查询包
            doc_body_query = {
                'query': {
                    "match": {
                        "name": data['text']
                    }
                },
                "size":es_size
            } 
            allDoc = self.es.search(index=self.indices, doc_type=self.doc_type, body=doc_body_query) 
            query2score_dict = {}
            candidate_query_list = []
            for doc in allDoc['hits']['hits']:
                # 对于 重复出现的，需要 加分
                if doc['_source']['name'] not in query2score_dict:
                    query2score_dict[doc['_source']['name']] = doc['_score']/(len(doc['_source']['name'])+len(data['text']))
                else:
                    query2score_dict[doc['_source']['name']] = doc['_score']/(len(doc['_source']['name'])+len(data['text']))*repeat_score

            # 候选 query 格式 转化
            for query,score in query2score_dict.items():
                candidate_query_list.append({
                    "query":query,
                    "score":score
                })
            candidate_query_list = sorted(candidate_query_list, key=lambda x: x['score'], reverse=True)[0:top]

            for candidate_query in candidate_query_list:
                if candidate_query['query'] in normalized_result_set:
                    hit = hit + 1 
            
            hit_rate = hit/len(normalized_result_set)
            result_dict_list.append({
                "query" : data['text'],
                "normalized_result":list(normalized_result_set),
                "candidate_query":candidate_query_list,
                "hit_rate":  hit_rate 
            })
            all_hit_score = all_hit_score+hit_rate
        
        print(f"all_hit_score/len(result_dict_list):{all_hit_score/len(result_dict_list)}")
        return result_dict_list

    # 功能：计算 用户 query 和 标准库 中 候选 query 的 ES 分数
    def get_documents_score(self, query, topN = 5):
        '''
            功能：计算 用户 query 和 标准库 中 候选 query 的 ES 分数
            input：
                query         String       用户输入 
                topN          int         选取 TopN
            return:
                candidate_query_score_list    Dict List 与 用户 query 相似的候选query和分数
        '''
        # es 查询包
        doc_body_query = {
            'query': {
                "match": {
                    "name": query
                }
            },
            "size":topN
        } 
        allDoc = self.es.search(index=self.indices, doc_type=self.doc_type, body=doc_body_query) 
        candidate_query_score_list = []
        for doc in allDoc['hits']['hits']:
            candidate_query_score_list.append({
                "query":doc['_source']['name'],
                "score":doc['_score']
            })
        return candidate_query_score_list
    
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
        # es 查询包
        doc_body_query = {
            'query': {
                "match": {
                    "name": query
                }
            },
            "size":topN
        } 
        allDoc = self.es.search(index=self.indices, doc_type=self.doc_type, body=doc_body_query) 
        candidate_query_list = []
        for doc in allDoc['hits']['hits']:
            candidate_query_list.append(doc['_source']['name'])
        return candidate_query_list

if __name__ == "__main__":
    # 变量设置
    origin_data_path = "F:/document/datasets/nlpData/CBLUE/CHIP-CDN/"
    data_path = "resource/"
    file_name = f"{origin_data_path}国际疾病分类 ICD-10北京临床版v601.xlsx"
    topN = 10
    file_name_list = [
        "CHIP-CDN_train","CHIP-CDN_dev"
    ]      
    # B25 模型定义
    es_model = ES_Model(indices = "cdn", doc_type = "cdn", host="127.0.0.1")
    
    # 数据导入 ES
    df = pd.read_excel(file_name,sheet_name='Sheet1',header=None)
    df.columns=['id','name']
    documents_list = list(df['name'])
    es_model.build_model(documents_list)

    # 数据加载
    data_list = load_json(f"{origin_data_path}{file_name_list[1]}.json")

    documents_score_list = es_model.get_documents_score(data_list[0]['text'],topN=topN)
    ''' 
        print(f"documents_score_list:{documents_score_list}")
        >>>
        documents_score_list:[{'query': '1型糖尿病性单神经病', 'score': 26.617285}, {'query': '1型糖尿病性股神经病', 'score': 26.617285}, {'query': '1型糖尿病性视网膜病变', 'score': 23.816156}, {'query': '2型糖尿病性单神经病', 'score': 22.672304}, {'query': '1型糖尿病性神经根病', 'score': 22.624714}, {'query': '2型糖尿病性股神经病', 'score': 22.463913}, {'query': '1型糖尿病', 'score': 22.265753}, {'query': '1型糖尿病性小神经纤维周围神经病', 'score': 22.142216}, {'query': '1型糖尿病性牙周炎', 'score': 22.135128}, {'query': '1型糖尿病性坏疽', 'score': 22.132566}]
    '''
    
    documents_list = es_model.get_candidate_query(data_list[0]['text'],topN=topN)
    ''' 
        print(f"documents_list:{documents_list}")
        >>>
        documents_list:['1型糖尿病性单神经病', '1型糖尿病性股神经病', '1型糖尿病性视网膜病变', '2型糖尿病性单神经病', '1型糖尿病性神经根病', '2型糖尿病性股神经病', '1型糖尿病', '1型糖尿病性小神经纤维周围神经病', '1型糖尿病性牙周炎', '1型糖尿病性坏 疽']
    '''


