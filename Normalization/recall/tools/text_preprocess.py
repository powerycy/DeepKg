from collections import defaultdict
import jieba
def getStopWord(stop_word_path='data/webMedQA/resource/stop_words.txt'):
    '''
        功能：获取停用词
        input:
            stop_word_path  String        停用词 路径 
        return:  
            stoplist        String List   停用词列表
    '''
    stop_word = [line.strip() for line in open(stop_word_path,encoding="utf-8").readlines() ]
    stoplist = list(set(stop_word))
    stoplist.append(" ")
    return stoplist

# 功能：数据分词处理
def cutWordDataProcess(documents,stoplist=[],freq=0,sep=" "):
    '''
        功能：数据分词处理
        input:
            documents       String List   文档信息 
            stoplist        String        停用词 列表
            frep:           int           频率
            sep：           String        分隔符
        return:  
            texts           String List   分词后的 文本
    '''
    # 分词，并清除 停用词
    documents = [[word for word in list(jieba.cut(document, cut_all=False)) if word not in stoplist] for document in documents]

    # 去掉只出现低于 freq 次的单词
    frequency = defaultdict(int)
    for text in documents:
        for token in text:
            frequency[token] += 1
    texts = []
    for text in documents:
        tokens = []
        for token in text:
            if frequency[token] > freq:
                tokens.append(token)
        texts.append(tokens)
    texts = [sep.join(text) for text in texts]
    return texts

# 功能：数据分字处理
def cutCharDataProcess(documents,stoplist=[],freq=0,sep=" "):
    '''
        功能：数据分字处理
        input:
            documents       String List   文档信息 
            stoplist        String        停用词 列表
            frep:           int           频率
            sep：           String        分隔符
        return:  
            texts           String List   分词后的 文本
    '''
    # 分词，并清除 停用词
    documents = [[char for char in list(document) if char not in stoplist] for document in documents]

    # 去掉只出现低于 freq 次的单词
    frequency = defaultdict(int)
    for text in documents:
        for token in text:
            frequency[token] += 1
    texts = []
    for text in documents:
        tokens = []
        for token in text:
            if frequency[token] > freq:
                tokens.append(token)
        texts.append(tokens)
    texts = [sep.join(text) for text in texts]
    return texts

# 功能：利用 n-gram 处理文本
def ngramDataProcess(documents,stoplist=[],freq=0,sep=" ",n=2):
    '''
        功能：利用 n-gram 处理文本
        input:
            documents       String List   文档信息 
            stoplist        String        停用词 列表
            frep:           int           频率
            sep：           String        分隔符
        return:  
            texts           String List   分词后的 文本
    '''
    documents = ["".join([word for word in document if word not in stoplist]) for document in documents]
    texts = []
    ngramLists = []
    for doc in documents:
        ngramList = []
        for i in range(0,len(doc)-n):
            if doc[i:i+n] not in stoplist:
                ngramList.append(doc[i:i+n])
            if doc[i] not in stoplist:
                ngramList.append(doc[i])
        if len(doc)-n>0:
            for i in range(len(doc)-n,len(doc)):
                ngramList.append(doc[i])
        ngramLists.append(ngramList)

    frequency = defaultdict(int)
    for text in ngramLists:
        for token in text:
            frequency[token] += 1
    texts = []
    for text in ngramLists:
        tokens = []
        for token in text:
            if frequency[token] > freq:
                tokens.append(token)
        texts.append(tokens)

    texts = [sep.join(text) for text in texts]
    return texts