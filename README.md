# 【知识图谱构建 DeepKg】 

## 一、前言

本项目致力于知识图谱的构建，目前正一点一点搭建其方法，也希望能帮助更多的人，<br>



## 二、复现模块

### 2.1 [【知识图谱构建 之 实体识别篇】](ExtractionEntities/)

【知识图谱构建 之 实体识别篇】 目前提供了一些torch版本的NER方法，包含GlobalPointer,Biaffine,UnlabeledEntity，以及R-drop的应用,EMA方法的应用。
具体细节请查看：https://zhuanlan.zhihu.com/p/375805722

### 2.2 [【知识图谱构建 之 语义标准化篇】](Normalization/)

【知识图谱构建 之 语义标准化篇】 目前采用 [召回（recall）](Normalization/recall/) 和 [排序（rank）](Normalization/rank/)方式进行 语义标准化，目前 只 提供  [召回（recall）](Normalization/recall/) 。
### 2.3 [【知识图谱构建 之 向量召回】](SimCSE-Chinese-Pytorch/)

【知识图谱构建 之 向量召回】 目前采用 SimCSE跟faiss搭配在一起，支持无监督跟有监督训练。
 详情请看苏神的解读：https://spaces.ac.cn/archives/8348
### 2.4 [【知识图谱构建 之 AutoPhrase】](AutoPhraseX-main/)

【知识图谱构建 之 专业词挖掘】 目前采用AutoPhrase非常经典跟实用的方法，并加入了bert来做。
 也感谢大佬提供的实现版本:https://github.com/luozhouyang/AutoPhraseX<br>
 详情请见知乎：https://zhuanlan.zhihu.com/p/434919516


