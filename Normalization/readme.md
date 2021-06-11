# 【知识图谱构建 之 语义标准化篇】

- [【知识图谱构建 之 语义标准化篇】](#知识图谱构建-之-语义标准化篇)
  - [一、前言](#一前言)
  - [二、所复现算法](#二所复现算法)
    - [2.1 介绍](#21-介绍)
    - [2.2 召回篇](#22-召回篇)
    - [2.3 排序篇](#23-排序篇)
  - [三、数据格式介绍](#三数据格式介绍)
  - [四、环境](#四环境)
  - [五、如何运行](#五如何运行)
    - [5.1 安装 python 包](#51-安装-python-包)
    - [5.2 召回篇](#52-召回篇)
    - [5.3 排序篇](#53-排序篇)
  - [六、运行结果](#六运行结果)
    - [6.1 召回篇](#61-召回篇)
      - [6.1.1 bm25 实验结果分析](#611-bm25-实验结果分析)
      - [6.1.2 es 实验结果分析](#612-es-实验结果分析)
    - [6.2 排序篇](#62-排序篇)
  - [参考资料](#参考资料)

## 一、前言

语义标准化作为信息抽取的一个重要子任务，近些年已经取得了阶段性成果。本项目主要对目前目前前沿的语义标准化算法进行复现。并在 【[临床术语标准化任务（CHIP-CDN）](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.22060218.J_2657303350.1.70e81343dFDilp&dataId=95414)】验证 复现算法 的性能。

## 二、所复现算法

### 2.1 介绍

1. 支持 multiprocessing 并行化；
2. 支持 BM2.5 召回
   1. 基于 字 的 BM2.5 召回
   2. 基于 词（采用 jieba分词 分词） 的 BM2.5 召回
   3. 基于 n-gram 的 BM2.5 召回
3. 支持 ES 召回

### 2.2 召回篇

- BM2.5 召回
- ES 召回

### 2.3 排序篇


## 三、数据格式介绍

对于给定的一组纯医学文本文档，任务的目标是针对中文电子病历中挖掘出的真实诊断实体进行语义标准化。 给定一诊断原词，要求给出其对应的诊断标准词。所有诊断原词均来自于真实医疗数据，并以《国际疾病分类 ICD-10 北京临床版v601》词表为标准进行了标注。标注样例如下（注：预测值可能存在多个，用“##”分隔）：

```s
[
  {
    "text": "左膝退变伴游离体",
    "normalized_result": "膝骨关节病##膝关节游离体"
  },
  {
    "text": "糖尿病反复低血糖;骨质疏松;高血压冠心病不稳定心绞痛",
    "normalized_result": "糖尿病性低血糖症##骨质疏松##高血压##冠状动脉粥样硬化性心脏病##不稳定性心绞痛"
  },
  {
    "text": "右乳腺癌IV期",
    "normalized_result": "乳腺恶性肿瘤##癌"
  },
  {
    "text": "头痛.头晕.高血压",
    "normalized_result": "头痛##头晕##高血压"
  },
  ...
]
```

## 四、环境

- python == 3.6+
- elasticsearch==7.10.1
- numpy==1.20.3
- jieba==0.39
- pandas==1.1.5
- rank-bm25==0.2.1
- scikit-learn==0.24.1
- scipy==1.5.4

## 五、如何运行

### 5.1 安装 python 包

```s
  pip install -r requirements.txt
```

### 5.2 召回篇

```s
  $ python inference.py --model_type=es --config_file=./train_config/config_altas.ini
  >>>
    函数：build_model() 共耗时约 0.25370 秒
  ~~~~~~~~~~multiprocessing start~~~~~~~~~~~~
  Prefix dict has been built succesfully.
  100%|████████████████████████████████████████████████████████████████████████████████████████████| 375/375 [04:43<00:00,  1.32it/s]
  375it [00:00, 33662.15it/s]
  Sub-process(es) done.
  ...
  100%|████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [09:02<00:00,  1.15it/s]
  Sub-process(es) done.
```

> 注：
> model_type  为 模型类型， es or bm25

### 5.3 排序篇

...

## 六、运行结果

### 6.1 召回篇

#### 6.1.1 bm25 实验结果分析

| v | cut_status |  topN  | simScore | data | 
| - | ---------- | ------ | -------- | ---- |  
| 1 |    ngram   |   100  | 0.498191 |  dev | 
| 1 |    ngram   |   500  | 0.604494 |  dev | 
| 2 |    char    |   100  | 0.330833 |  dev | 
| 2 |    char    |   500  | 0.471883 |  dev | 
| 3 |     cut    |    50  | 0.606243 |  dev | 
| 3 |     cut    |   100  | 0.654615 |  dev | 
| 3 |     cut    |   500  | 0.757890 |  dev | 

#### 6.1.2 es 实验结果分析

| v | cut_status |  topN  | simScore | data | 
| - | ---------- | ------ | -------- | ---- |  

### 6.2 排序篇

|        模型      |  acc  | precision | recall |  val_f_beta  |
|       ----       | ----- | --------- | ------ |   --------   |

## 参考资料

1. [elastic 介绍](https://www.elastic.co/cn/?ultron=B-Stack-Trials-AMER-US-W-Exact&gambit=Elasticsearch-Core&blade=adwords-s&hulk=cpc&Device=c&thor=elasticsearch&gclid=Cj0KCQjwk4yGBhDQARIsACGfAeuTzXw-2v2GBrgF_-nouxlLnZbrpunIS5waP_oKCVlmkm3bCiazUNkaAqfnEALw_wcB)
2. [bm2.5 介绍](https://pypi.org/project/rank-bm25/)


