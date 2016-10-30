import json
import logging
import os
import random
from copy import deepcopy
import torch
from sklearn.ensemble import GradientBoostingClassifier
# import lightgbm as lgb
from transformers import BertTokenizerFast,BertModel
# from model import BertForSequenceClassification
from autophrasex.extractors import FeatureExtractorWrapper
from autophrasex.reader import AbstractCorpusReader
import torch.nn as nn
from . import utils
from .callbacks import CallbackWrapper, StateCallback
from .selector import AbstractPhraseSelector
model_path='/home/yuanchaoyi/BeiKe/QA_match/roberta_base'
def load_quality_phrase_files(input_files):
    pharses = set()

    def collect_fn(line, lino):
        pharses.add(line.strip())

    utils.load_input_files(input_files, callback=collect_fn)
    return pharses

def concat_pos(x):
    return x
class AutoPhrase:

    def __init__(self,
                 reader: AbstractCorpusReader,
                 selector: AbstractPhraseSelector,
                 extractors=None,
                 classifier=None,
                 threshold=0.4,
                 **kwargs):
        """Constractor

        Args:
            reader: Instance of AbstractCorpusReader, used to read corpus files
            selector: Instance of AbstractPhraseSelector, used to select frequent phrases
            extractors: List of AbstractFeatureExtractor, used to extract features for classifier
            threshold: Python float, negative phrase whose prob greater than this will be moved to positive pool
        """
        self.selector = selector
        self.extractors = extractors or []
        self.extractor_wrapper = FeatureExtractorWrapper(extractors=self.extractors)
        self.corpus_reader = reader
        # self.embedding = nn.Embedding(3,768)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.bert = BertModel.from_pretrained(model_path)
        if classifier is None:
            # classifier = RandomForestClassifier(**kwargs)
            classifier = GradientBoostingClassifier(**kwargs)
        self.classifier = classifier

        # used by ThresholdSchedule
        self.threshold = threshold
        # used by EarlyStopping
        self.early_stop = False

    def mine(self,
             corpus_files,
             quality_phrase_files,
             N=4,
             epochs=10,
             callbacks=None,
             topk=1000,
             filter_fn=None,
             **kwargs):
        """Mining phrase from corpus.

        Args:
            corpus_files: Files of corpus
            quality_phrase_files: File path(s) of quality phrases, one phrase each line
            epochs: Python integer, Number of training epoch
            callbacks: List of Callback, used to listen lifecycles
            topk: Python integer, Number of frequent phrases selected from Selector
            filter_fn: Python callable, with signature fn(phrase, freq), used to filter phrases

        Return:
            predictions: List of tuple (phrase, prob), predict from initial negative phrase pool
        """
        callback = CallbackWrapper(callbacks=callbacks)
        callback.begin()

        # setup state callbacks
        for cb in callbacks:
            if isinstance(cb, StateCallback):
                cb.autophrase = self

        callback.on_read_corpus_begin()
        self.corpus_reader.read(
            corpus_files,
            self.extractor_wrapper,
            N=N,
            verbose=kwargs.get('verbose', True),
            logsteps=kwargs.get('logsteps', 1000))
        callback.on_read_corpus_end()

        callback.on_build_quality_phrases_begin(quality_phrase_files)
        quality_phrases = load_quality_phrase_files(quality_phrase_files)
        callback.on_build_quality_phrases_end(quality_phrases)

        callback.on_select_frequent_phrases_begin()
        frequent_phrases = self.selector.select(
            extractors=self.extractors,
            topk=topk,
            filter_fn=filter_fn,
            **kwargs)
        callback.on_select_frequent_phrases_end(frequent_phrases)

        callback.on_organize_phrase_pools_begin(x, frequent_phrases)
        pos_pool, neg_pool,initial_pos_token, initial_pos_mask,initial_neg_token, initial_neg_mask,initial_pos_type, initial_neg_type = self._organize_phrase_pools(quality_phrases, frequent_phrases, **kwargs) #找到正立与负例样本
        callback.on_organize_phrase_pools_end(pos_pool, neg_pool)
        # pos_pool, neg_pool = initial_pos_pool, initial_neg_pool
        pos_bert = self.bert(initial_pos_token,initial_pos_mask,initial_pos_type).pooler_output
        neg_bert = self.bert(initial_neg_token,initial_neg_mask,initial_neg_type).pooler_output
        pos_pool = list(map(concat_pos,zip(pos_pool,pos_bert.tolist())))
        neg_pool = list(map(concat_pos,zip(neg_pool,neg_bert.tolist())))
        initial_neg_pool = neg_pool
        for epoch in range(epochs):
            callback.on_epoch_begin(epoch)

            callback.on_epoch_prepare_training_data_begin(epoch)
            x, y = self._prepare_training_data(pos_pool, neg_pool, **kwargs)
            callback.on_epoch_prepare_training_data_end(epoch, x, y)

            self.classifier.fit(x, y)

            callback.on_epoch_reorganize_phrase_pools_begin(epoch, pos_pool, neg_pool)
            pos_pool, neg_pool = self._reorganize_phrase_pools(pos_pool, neg_pool, **kwargs)
            callback.on_epoch_reorganize_phrase_pools_end(epoch, pos_pool, neg_pool)

            if self.early_stop:
                logging.info('early stop!')
                break

            callback.on_epoch_end(epoch)

        callback.on_predict_neg_pool_begin(neg_pool)
        predictions = self._predict_proba(initial_neg_pool)
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
        callback.on_predict_neg_pool_end(predictions)

        callback.end()
        return predictions

    def _prepare_training_data(self, pos_pool, neg_pool, **kwargs):
        x, y = [], []
        examples = []
        for p in pos_pool:
            feature_value = self._compose_feature(p[0])
            feature_value.extend(p[1])
            examples.append((feature_value, 1))
        for p in neg_pool:
            feature_value = self._compose_feature(p[0])
            feature_value.extend(p[1])
            examples.append((feature_value, 0))
        # shuffle
        random.shuffle(examples)
        for _x, _y in examples:
            x.append(_x)
            y.append(_y)
        return x, y

    def _reorganize_phrase_pools(self, pos_pool, neg_pool, **kwargs):
        new_pos_pool, new_neg_pool = [], []
        new_pos_pool.extend(deepcopy(pos_pool))

        pairs = self._predict_proba(neg_pool)
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        # print(pairs[:10])

        for _, (p, prob) in enumerate(pairs):
            if prob > self.threshold:
                new_pos_pool.append(p)
                continue
            new_neg_pool.append(p)

        return new_pos_pool, new_neg_pool

    def _organize_phrase_pools(self, quality_phrases, frequent_phrases, **kwargs):
        pos_pool, neg_pool = [], []
        pos_token,neg_token = [],[]
        pos_token_mask,neg_token_mask = [],[]
        pos_token_type,neg_token_type = [],[]
        for p in frequent_phrases:
            if p in quality_phrases:
                token = self.tokenizer(p,return_offsets_mapping=True,max_length=5,truncation=True,padding='max_length',return_tensors='pt')
                pos_pool.append(p)
                pos_token.append(token['input_ids'])
                pos_token_mask.append(token['attention_mask'])
                pos_token_type.append(token['token_type_ids'])
                continue
            _p = ''.join(p.split(' '))
            if _p in quality_phrases:
                token = self.tokenizer(_p,return_offsets_mapping=True,max_length=5,truncation=True,padding='max_length',return_tensors='pt')
                pos_pool.append(p)
                pos_token.append(token['input_ids'])
                pos_token_mask.append(token['attention_mask'])
                pos_token_type.append(token['token_type_ids'])
                continue
            neg_pool.append(p)
            token = self.tokenizer(p,return_offsets_mapping=True,max_length=5,truncation=True,padding='max_length',return_tensors='pt')
            neg_token.append(token['input_ids'])
            neg_token_mask.append(token['attention_mask'])
            neg_token_type.append(token['token_type_ids'])
        pos_token = torch.cat([item for item in pos_token ],dim=0)
        pos_token_mask = torch.cat([item for item in pos_token_mask],dim=0)   
        neg_token = torch.cat([item for item in neg_token],dim=0)
        neg_token_mask = torch.cat([item for item in neg_token_mask],dim=0) 
        pos_token_type = torch.cat([item for item in pos_token_type ],dim=0)
        neg_token_type = torch.cat([item for item in neg_token_type ],dim=0)
        return pos_pool, neg_pool,pos_token,pos_token_mask,neg_token,neg_token_mask,pos_token_type,neg_token_type

    def _predict_proba(self, phrases):
        features_bert = []
        # features = [self._compose_feature(phrase) for phrase in phrases[0]]
        for phrase in phrases:
            features = self._compose_feature(phrase[0])
            features.extend(phrase[1])
            features_bert.append(features)
        # self.classifier.predict_proba(features)
        pos_probs = [prob[1] for prob in self.classifier.predict_proba(features_bert)]
        pairs = [(phrase, prob) for phrase, prob in zip(phrases, pos_probs)]
        return pairs

    def _compose_feature(self, phrase):
        features = self.extractor_wrapper.extract(phrase)
        features = sorted(features.items(), key=lambda x: x[0])
        features = [x[1] for x in features]
        return features
