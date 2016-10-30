import torch
# a = ['a','b','c','d']
# b = ['e','f','g','h']
# def concat(x):
#     z = x
#     return z
# print(list(map(concat,zip(a,b))))
import jieba
a = jieba.lcut('马应龙痔疮膏')
print(a)