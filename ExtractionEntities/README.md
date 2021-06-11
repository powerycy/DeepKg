# 【知识图谱构建 之 实体识别篇】

- [【知识图谱构建 之 实体识别篇】](#知识图谱构建-之-实体识别篇)
  - [一、前言](#一前言)
  - [二、所复现算法](#二所复现算法)
    - [2.1 算法介绍](#21-算法介绍)
    - [2.2 支持功能介绍](#22-支持功能介绍)
  - [三、数据格式介绍](#三数据格式介绍)
  - [四、环境](#四环境)
  - [五、如何运行](#五如何运行)
    - [5.1 安装 python 包](#51-安装-python-包)
    - [5.2 模型训练](#52-模型训练)
    - [5.3 模型预测](#53-模型预测)
    - [5.4 模型DDP多卡形式训练](#54-模型ddp多卡形式训练)
    - [5.5 模型DDP多卡形式预测](#55-模型ddp多卡形式预测)
  - [六、运行结果](#六运行结果)
  - [参考资料](#参考资料)

## 一、前言

实体识别作为信息抽取的一个重要子任务，近些年已经取得了阶段性成果。本项目主要对目前目前前沿的实体识别算法进行复现。并在 【[中文医学命名实体识别（CMeEE）](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.22060218.J_2657303350.1.70e81343dFDilp&dataId=95414)】验证 复现算法 的性能。

## 二、所复现算法

### 2.1 算法介绍

- [【Unlabeled Entity Problem】Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2012.05426)
- [【Biaffine Ner】Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition](https://www.aclweb.org/anthology/2020.acl-main.577/)
- [【GlobalPointer】](https://spaces.ac.cn/archives/8373)

### 2.2 支持功能介绍

1. 支持 DDP多卡形式训练；
2. 支持 日志打印；

## 三、数据格式介绍

对于给定的一组纯医学文本文档，任务的目标是识别并抽取出与医学临床相关的实体，并将他们归类到预先定义好的类别。将医学文本命名实体划分为九大类，包括：疾病(dis)，临床表现(sym)，药物(dru)，医疗设备(equ)，医疗程序(pro)，身体(bod)，医学检验项目(ite)，微生物类(mic)，科室(dep)。标注之前对文章进行自动分词处理，所有的医学实体均已正确切分。

标注的“输入/输出”示例:

```s
输入:句子sentence。

输出: 句子sentence中包含的医学实体位置（注：起始、结束下标采用左闭右闭的记法）和实体类型。

{  
  "text": "呼吸肌麻痹和呼吸中枢受累患者因呼吸不畅可并发肺炎、肺不张等。", 
  "entities": [ 
    { 
      "start_idx": 0, 
      "end_idx": 2, 
      "type": "bod", 
      "entity: "呼吸肌" 
    }, 
    { 
      "start_idx": 0, 
      "end_idx": 4, 
      "type": "sym",
       "entity: "呼吸肌麻痹" 
     }, 
     { 
       "start_idx": 6, 
       "end_idx": 9,
       "type": "bod", 
       "entity: "呼吸中枢"
     }, 
     { 
       "start_idx": 6, 
       "end_idx": 11, 
       "type": "sym", 
       "entity: "呼吸中枢受累" 
   }, 
   { 
      "start_idx": 15, 
      "end_idx": 18, 
      "type": "sym", 
      "entity: "呼吸不畅" 
    }, 
   { 
      "start_idx": 22, 
      "end_idx": 23, 
      "type": "dis", 
      "entity: "肺炎" 
    }, 
   { 
      "start_idx": 25, 
      "end_idx": 27, 
      "type": "dis", 
      "entity: "肺不张" 
    } 
  ] 
}

```

## 四、环境

- python == 3.6+
- transformers == 4.1.1
- torch == 1.7.0
- torchaudio==0.7.0
- torchvision==0.8.1
- tqdm==4.61.0
- numpy==1.20.3

## 五、如何运行

### 5.1 安装 python 包

```s
  pip install -r requirements.txt
```

### 5.2 模型训练

```s
    $ python main.py --model_type=biaffine --config_file=./train_config/config_yang.ini
    >>>
    [2021-06-10 20:58:45]: ner main.py[line:170] INFO  Epoch 1
    [2021-06-10 20:58:56]: ner main.py[line:123] INFO  loss: 6.610330  [    0/15000]
    ...
    [2021-06-11 09:22:04]: ner main.py[line:123] INFO  loss: 0.291933  [14400/15000]
    [2021-06-11 09:26:37]: ner main.py[line:123] INFO  loss: 0.305781  [14800/15000]
    [2021-06-11 09:28:48]: ner main.py[line:124] INFO  Train F1: 0.748511%
    [2021-06-11 09:46:01]: ner main.py[line:144] INFO  Test Error:
    ,F1:0.648233,Avg loss: 0.081294

    [2021-06-11 09:46:01]: ner main.py[line:159] INFO  valid:  f1: 0.64823,  best f1: 0.64861

```

> 参数介绍：
> model_type     实体识别模型    ['biaffine', 'UnlabeledEntity', 'globalpointer']
> config_file    可配置所需的参数

### 5.3 模型预测

```s
  $ python inference.py --model_type=UnlabeledEntity --config_file=./train_config/config_altas.ini
  >>>
  define model!
  loading model!
  save_model/multilabel_glob_UnlabeledEntity.pth
  100%|███████████████████████████████████████████████████████████| 3000/3000 [11:02<00:00,  4.53it/s]
```

### 5.4 模型DDP多卡形式训练

```s
    $ python main_ddp.py --model_type=biaffine --config_file=./train_config/config_yang.ini
    >>>
    [2021-06-10 20:58:45]: ner main.py[line:170] INFO  Epoch 1
    [2021-06-10 20:58:56]: ner main.py[line:123] INFO  loss: 6.610330  [    0/15000]
    ...
    [2021-06-11 09:22:04]: ner main.py[line:123] INFO  loss: 0.291933  [14400/15000]
    [2021-06-11 09:26:37]: ner main.py[line:123] INFO  loss: 0.305781  [14800/15000]
    [2021-06-11 09:28:48]: ner main.py[line:124] INFO  Train F1: 0.748511%
    [2021-06-11 09:46:01]: ner main.py[line:144] INFO  Test Error:
    ,F1:0.648233,Avg loss: 0.081294

    [2021-06-11 09:46:01]: ner main.py[line:159] INFO  valid:  f1: 0.64823,  best f1: 0.64861
```

### 5.5 模型DDP多卡形式预测

```s
  $ python inference_ddp.py --model_type=UnlabeledEntity --config_file=./train_config/config_altas.ini
  >>>
  define model!
  loading model!
  save_model/multilabel_glob_UnlabeledEntity.pth
  100%|███████████████████████████████████████████████████████████| 3000/3000 [11:02<00:00,  4.53it/s]
```

## 六、运行结果

|        模型      |  acc  | precision | recall |  val_f_beta  |
|       ----       | ----- | --------- | ------ |   --------   |
|     biaffine     | ----- | --------- | ------ |   0.63930    |
| UnlabeledEntity  | ----- | --------- | ------ |   0.64861    |



## 参考资料

1. [Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2012.05426)
2. [Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition](https://www.aclweb.org/anthology/2020.acl-main.577/)
3. [GlobalPointer](https://spaces.ac.cn/archives/8373)


