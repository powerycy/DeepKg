import os
import unittest2

import jieba
from autophrasex import utils
from autophrasex.autophrase import AutoPhrase
from autophrasex.callbacks import (ConstantThresholdScheduler, EarlyStopping,
                                   LoggingCallback)
from autophrasex.extractors import (EntropyExtractor, IDFExtractor,
                                    NgramsExtractor)
from autophrasex.reader import DefaultCorpusReader
from autophrasex.selector import DefaultPhraseSelector
from autophrasex.tokenizer import JiebaTokenizer


class AutoPhraseTest(unittest2.TestCase):

    def test_autophrase_small(self):
        N = 4
        autophrase = AutoPhrase(
            reader=DefaultCorpusReader(tokenizer=JiebaTokenizer()), #初始化Jieba
            selector=DefaultPhraseSelector(min_len=3),
            extractors=[NgramsExtractor(N=N), IDFExtractor(), EntropyExtractor()]
        )

        predictions = autophrase.mine(
            corpus_files=['../data/zh_china.txt'],
            quality_phrase_files='../CN/wiki_quality.txt',
            N=N,
            callbacks=[
                LoggingCallback(),
                ConstantThresholdScheduler(),
                EarlyStopping(patience=1, min_delta=3)
            ])
        for pred in predictions:
            print(pred)


if __name__ == "__main__":
    unittest2.main()
