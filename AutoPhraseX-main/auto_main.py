from autophrasex import *
from autophrasex.tokenizer import AbstractTokenizer
from autophrasex.selector import DefaultPhraseSelector
from autophrasex.extractors import AbstractNgramFiter
from  autophrasex.reader import AbstractCorpusReadCallback
# from test import slide_word
# from LAC import LAC
#
# class BaiduLacTokenizer(AbstractTokenizer):
#
#     def __init__(self, custom_vocab_path=None, model_path=None, mode='seg', use_cuda=False, **kwargs):
#         self.lac = LAC(model_path=model_path, mode=mode, use_cuda=use_cuda)
#         if custom_vocab_path:
#             self.lac.load_customization(custom_vocab_path)
#
#     def tokenize(self, text, **kwargs):
#         text = self._uniform_text(text, **kwargs)
#         results = self.lac.run(text)
#         results = [x.strip() for x in results if x.strip()]
#         return results
#
# class VerbPhraseFilter(AbstractPhraseFilter):
#
#     def __init__(self, batch_size=100):
#         super().__init__()
#         self.lac = LAC()
#         self.batch_size = batch_size
#
#     def batch_apply(self, batch_pairs, **kwargs):
#         predictions = []
#         for i in range(0, len(batch_pairs), self.batch_size):
#             batch_texts = [x[0] for x in batch_pairs[i: i + self.batch_size]]
#             batch_preds = self.lac.run(batch_texts)
#             predictions.extend(batch_preds)
#         candidates = []
#         for i in range(len(predictions)):
#             _, pos_tags = predictions[i]
#             if any(pos in ['v', 'vn', 'vd'] for pos in pos_tags):
#                 continue
#             candidates.append(batch_pairs[i])
#         return candidates
#
# selector = DefaultPhraseSelector(
#     phrase_filters=[VerbPhraseFilter()],
#     use_default_phrase_filters=False
# )
# CHARACTERS = set('!"#$%&\'()*+,-./:;?@[\\]^_`{|}~ \t\n\r\x0b\x0c，。？：“”【】「」')
#
# class MyNgramFilter(AbstractNgramFiter):
#
#     def apply(self, ngram, **kwargs):
#         if any(x in CHARACTERS for x in ngram):
#             return True
#         return False
# class UnigramFeatureExtractor(AbstractFeatureExtractor,AbstractCorpusReadCallback):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def extract(self, phrase, **kwargs):
#         parts = phrase.split(' ')
#         features = {
#             'is_unigram': 1 if len(parts) == 1 else 0
#         }
#         return features
#
# autophrase = AutoPhrase(
#     reader = DefaultCorpusReader(tokenizer=BaiduLacTokenizer()),
#     selector=DefaultPhraseSelector(
#         phrase_filters=[VerbPhraseFilter()],
#         use_default_phrase_filters=False,
#     ),
#     extractors=[
#         NgramsExtractor(N=4, ngram_filters=[MyNgramFilter()]),
#         IDFExtractor(ngram_filters=[MyNgramFilter()]),
#         EntropyExtractor(ngram_filters=[MyNgramFilter()]),
#         UnigramFeatureExtractor(),
#     ]
# )
#
# # # 开始挖掘
# predictions = autophrase.mine(
#     corpus_files=['./data/zh_china.txt'],
#     quality_phrase_files='./CN/wiki_quality.txt',
#     callbacks=[
#         LoggingCallback(),
#         ConstantThresholdScheduler(),
#         EarlyStopping(patience=2, min_delta=3)
#     ])
# # 输出挖掘结果
# for pred in predictions:
#     print(pred)
N=8
class NgramTokenizer(AbstractTokenizer):
    def tokenize(self,text,**kwargs):
        text = self._uniform_text(text,**kwargs)
        result = list(text)
        return result
# # 构造autophrase
autophrase = AutoPhrase(threshold=0.5,
    reader=DefaultCorpusReader(tokenizer=JiebaTokenizer('./data/医疗_vocab.txt')),
    selector=DefaultPhraseSelector(min_len=3),
    extractors=[
        NgramsExtractor(N=N),
        IDFExtractor(),
        EntropyExtractor()
    ]
)

# 开始挖掘
predictions = autophrase.mine(
    corpus_files=['ans.txt'],
    quality_phrase_files='./data/医疗_vocab.txt',
    callbacks=[
        LoggingCallback(),
        ConstantThresholdScheduler(),
        EarlyStopping(patience=2, min_delta=3)
    ])

# 输出挖掘结果
for pred in predictions:
    print(pred[0][0],pred[1])