# 【知识图谱构建 之 实体识别篇】

## 一、前言

实体识别作为信息抽取的一个重要子任务，近些年已经取得了阶段性成果。本项目主要对目前目前前沿的实体识别算法进行复现。并在 【[中文医学命名实体识别（CMeEE）](https://tianchi.aliyun.com/dataset/dataDetail?spm=5176.22060218.J_2657303350.1.70e81343dFDilp&dataId=95414)】验证 复现算法 的性能。

## 二、所复现算法

- [【Unlabeled Entity Problem】Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2012.05426)
- [【Biaffine Ner】Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition](https://www.aclweb.org/anthology/2020.acl-main.577/)
- [【GlobalPointer】](https://spaces.ac.cn/archives/8373)


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

## 五、如何运行

```s
    python main.py --model_type=biaffine --config_file=./train_config/config_yang.ini
```

> 参数介绍：
> model_type     实体识别模型    ['biaffine', 'UnlabeledEntity', 'globalpointer']
> config_file    可配置所需的参数


1. UnlabeledEntity和Biaffine直接运行就可以，inference可以预测出结果
2. 分别对应三个模型Globalpointer为DDP多卡形式，运行脚本run_globalpointer即可，运行ddp_inference或者inference都可得到结果。

## 六、运行结果

|        模型      |  acc  | precision | recall |  val_f_beta  |
|       ----       | ----- | --------- | ------ |   --------   |
|     biaffine     | ----- | --------- | ------ |   0.63930    |
| UnlabeledEntity  | ----- | --------- | ------ |   0.64861    |




## 参考资料

1. [Named Entity Recognition as Dependency Parsing](https://arxiv.org/abs/2012.05426)
2. [Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition](https://www.aclweb.org/anthology/2020.acl-main.577/)
3. [GlobalPointer](https://spaces.ac.cn/archives/8373)


