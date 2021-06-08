## 如何运行
1. 分别对应三个模型Globalpointer,UnlabeledEntity和Biaffine模型,可以运行阿里的医疗大赛数据
2. config.ini文件可配置所需的参数
3. UnlabeledEntity和Biaffine直接运行就可以，inference可以预测出结果
4. Globalpointer为DDP多卡形式，运行脚本run_globalpointer即可，运行ddp_inference或者inference都可得到结果。
5. 数据集地址:https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
