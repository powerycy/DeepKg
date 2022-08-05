import jieba
sen = '标准间太差房间还不如3星的而且设施非常陈旧.建议酒店把老的标准间从新改善.'
words = jieba.lcut(sen)
for i in range(len(words)):
    sentence_train = "".join(words[:i])+"，酒店[MASK]，"+"".join(words[i:])
    print(sentence_train)
