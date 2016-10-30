from importlib_metadata import re
import jieba

text = '去北京大学玩'
res = jieba.lcut(text)

print(res)
