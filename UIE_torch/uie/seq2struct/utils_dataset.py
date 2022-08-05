
# Copyright (c) 2021 DataArk Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Chaoyi Yuan, chaoyiyuan3721@gmail.com
# Status: Active
import inspect
import warnings
# from utils_torch import read_training_instance_based_config
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from transformers import T5ForConditionalGeneration
from uie.seq2struct.t5tokenizer import T5BertTokenizer
from Config import Config
config_path = 'config.ini'
config = Config(config_path)
tokenizer = T5BertTokenizer.from_pretrained(config.model_path)
model = T5ForConditionalGeneration.from_pretrained(config.model_path)
def _read(filename,*args):
    raise NotImplementedError
    # return None
def get_vocab():
    """
    Returns vocab file path of the dataset if specified.
    """
    return None
label_list = []
vocab_info = get_vocab()
class MapDataset():
    """
    Wraps a map-style dataset-like object as an instance of `MapDataset`, and equips it 
    with `map` and other utility methods. All non-magic methods of the raw object
    are also accessible.

    Args:
        data (list|Dataset): An object with `__getitem__` and `__len__` methods. It could 
            be a list or a subclass of `paddle.io.Dataset`.
        kwargs (dict, optional): Other information to be passed to the dataset. 

    For examples of this class, please see `dataset_self_defined 
    <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html>`__.

    """

    def __init__(self, data, **kwargs):
        self.data = data
        self._transform_pipline = []
        self.new_data = self.data
        self.info = kwargs
        self.label_list = self.info.pop('label_list', None)
        self.vocab_info = self.info.pop('vocab_info', None)

    def _transform(self, data):
        for fn in self._transform_pipline:
            data = fn(data)
        return data

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given 
        index.
        """
        return self._transform(self.new_data[
            idx]) if self._transform_pipline else self.new_data[idx]

    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.new_data)
class SimpleBuilder():
    def __init__(self,read_func):
        self._read = read_func
    def read(self, **kwargs):
        examples = self._read(**kwargs)
        if hasattr(examples, '__len__') and hasattr(examples,
                                                    '__getitem__'):
            return MapDataset(examples)
        else:
            return MapDataset(list(examples))
def load_dataset(path_or_read_func,
                 name=None,
                 data_files=None,
                 splits=None,
                 **kwargs):
    """
    This method will load a dataset, either form PaddleNLP library or from a 
    self-defined data loading script, by calling functions in `DatasetBuilder`.

    For all the names of datasets in PaddleNLP library, see here:  `dataset_list 
    <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_list.html>`__.

    Either `splits` or `data_files` must be specified.

    Args:
        path_or_read_func (str|callable): Name of the dataset processing script 
            in PaddleNLP library or a custom data reading function.
        name (str, optional): Additional name to select a more specific dataset.
            Defaults to None.
        data_files (str|list|tuple|dict, optional): Defining the path of dataset
            files. If None. `splits` must be specified. Defaults to None.
        splits (str|list|tuple, optional): Which split of the data to load. If None.
            `data_files` must be specified. Defaults to None.
        lazy (bool, optional): Weather to return `MapDataset` or an `IterDataset`.
            True for `IterDataset`. False for `MapDataset`. If None, return the 
            default type of this dataset. Defaults to None.
        kwargs (dict): Other keyword arguments to be passed to the `DatasetBuilder`.

    Returns:
        A `MapDataset` or `IterDataset` or a tuple of those.

    For how to use this function, please see `dataset_load 
    <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_load.html>`__
    and `dataset_self_defined 
    <https://paddlenlp.readthedocs.io/zh/latest/data_prepare/dataset_self_defined.html>`__

    """
    # list1 = []
    if inspect.isfunction(path_or_read_func):
        # assert lazy is not None, "lazy can not be None in custom mode."
        kwargs['name'] = name
        kwargs['data_files'] = data_files
        kwargs['splits'] = splits
        custom_kwargs = {}
        for name in inspect.signature(path_or_read_func).parameters.keys():
            if name in kwargs.keys():
                custom_kwargs[name] = kwargs[name]

        reader_instance = SimpleBuilder(read_func=path_or_read_func)
        return reader_instance.read(**custom_kwargs)