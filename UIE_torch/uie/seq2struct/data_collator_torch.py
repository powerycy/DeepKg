
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
import numpy as np
from typing import Optional
import logging

import tokenizers
from uie.evaluation.constants import (
    BaseStructureMarker,
    text_start,
    spot_prompt,
    asoc_prompt,
    null_span,
)
import torch
from torch.utils.data import Dataset,DataLoader
from collections import OrderedDict
import random
import math
from uie.evaluation.sel2record import (
    RecordSchema, convert_spot_asoc)
logger = logging.getLogger("__main__")
import copy
from torch.nn import functional as F
class SpotAsocNoiser:
    spot_noise_ratio: float = 0.  # Ratio of insert spot not in raw record
    asoc_noise_ratio: float = 0.  # Ratio of insert asoc not in raw record
    null_span: str = null_span  # Null span string

    def random_insert_spot(self, spot_asoc, spot_label_list=None):
        """ Insert negative spot in random, sample negative spot from spot_label_list
        """
        # If no negative spot_label_list, skip insertion of spot null span
        if spot_label_list is None or len(spot_label_list) == 0:
            return spot_asoc

        random_num = sum(
            np.random.binomial(1, self.spot_noise_ratio, len(spot_asoc)))
        for _ in range(random_num):
            random_position = np.random.randint(low=0, high=len(spot_asoc))
            to_insert_negative_spot = {
                "span": self.null_span,
                "label":
                np.random.choice(spot_label_list
                                 ),  # Sample negative spot from spot_label_list
                'asoc': list()
            }
            spot_asoc.insert(random_position, to_insert_negative_spot)
        return spot_asoc

    def random_insert_asoc(self, spot_asoc, asoc_label_list=None):
        """ Insert negative asoc in random, sample negative asoc from asoc_label_list
        """
        # If no negative asoc_label_list, skip insertion of asoc null span
        if asoc_label_list is None or len(asoc_label_list) == 0:
            return spot_asoc

        spot_sum = len(spot_asoc)
        random_num = sum(np.random.binomial(1, self.asoc_noise_ratio, spot_sum))
        for _ in range(random_num):
            random_label = np.random.choice(asoc_label_list)
            spot_position = np.random.randint(low=0, high=len(spot_asoc))
            asoc_position = np.random.randint(
                low=0, high=len(spot_asoc[spot_position]['asoc']) + 1)
            # Insert random negative span at `asoc_position`
            spot_asoc[spot_position]['asoc'].insert(
                asoc_position, (random_label, self.null_span))
        return spot_asoc

    def add_noise(self, spot_asoc, spot_label_list, asoc_label_list):
        """ Add noise to target spot-asoc structure
        spot_asoc: raw spot-asoc structure
        spot_label_list: negative spot candidates
        asoc_label_list: negative asoc candidates
        """
        spot_asoc = self.random_insert_asoc(spot_asoc, asoc_label_list)
        spot_asoc = self.random_insert_spot(spot_asoc, spot_label_list)
        return spot_asoc
class DynamicSSIGenerator():
    """
    Sample negative spot and asoc to construct SSI
    """

    def __init__(self,
                 tokenizer,
                 schema: RecordSchema,
                 positive_rate=1,
                 negative=-1,
                 ordered_prompt=False) -> None:
        self.spot_dict = self.get_ordered_dict(schema.type_list, tokenizer) #spo2id
        self.asoc_dict = self.get_ordered_dict(schema.role_list, tokenizer)
        self.spot_list = list(self.spot_dict.keys())
        self.asoc_list = list(self.asoc_dict.keys())
        self.spot_prompt_id = tokenizer.convert_tokens_to_ids(spot_prompt) #额外特殊的字符
        self.asoc_prompt_id = tokenizer.convert_tokens_to_ids(asoc_prompt)
        self.text_start_id = tokenizer.convert_tokens_to_ids(text_start)
        self.positive_rate = positive_rate if 0 < positive_rate < 1 else 1
        self.negative = negative
        self.ordered_prompt = ordered_prompt
        logger.info(f"Meta Sample "
                    f"Negative: {self.negative}, "
                    f"Ordered SSI: {self.ordered_prompt}")

    @staticmethod
    def get_ordered_dict(schema_name_list, tokenizer):
        """ Get schema name -> id dict
        schema_name_list: ["人物", "组织机构"]
        """
        schema_ordered_dict = OrderedDict()
        for name in schema_name_list:
            # tokenizer.encode("人物") -> [8, 122]
            encoded_name = tokenizer.encode_plus(name,
                                            add_special_tokens=False,
                                            return_token_type_ids=None,
                                            truncation=True)
            schema_ordered_dict[name] = encoded_name["input_ids"]
        return schema_ordered_dict

    @staticmethod
    def sample_negative(postive, candidates, k=5):
        if k < 0:
            k = len(candidates)
        negative_set = set()
        for index in np.random.permutation(len(candidates))[:k].tolist():
            negative = candidates[index]
            if negative not in postive:
                negative_set.add(negative)

        return list(negative_set)

    def sample_spot(self, positive, candidates=None):
        """ Sample spot

        Args:
            positive (List[str]): Positive Spot List

        Returns:
            List[int]: spot index list
            List[str]: Sampled Positive Spot List
            List[str]: Sampled Negative Spot List
        """
        neg_cands = candidates if candidates is not None else self.spot_list

        negative_spot = self.sample_negative(postive=positive,
                                             candidates=neg_cands,
                                             k=self.negative)
        positive_spot = random.sample(
            positive, math.floor(len(positive) * self.positive_rate))

        converted_spot_prefix = self.convert_prefix(
            candidates=positive_spot + negative_spot,
            prompt=self.spot_prompt_id,
            mapper=self.spot_dict,
            ordered_prompt=self.ordered_prompt,
        )

        return converted_spot_prefix, positive_spot, negative_spot

    def sample_asoc(self, positive, candidates=None):
        """ Sample Asoc

        Args:
            positive (List[str]): Positive Asoc List

        Returns:
            List[int]: asoc index list
            List[str]: Sampled Negative Asoc List
        """
        neg_cands = candidates if candidates is not None else self.asoc_list
        negative_asoc = self.sample_negative(postive=positive,
                                             candidates=neg_cands,
                                             k=self.negative)
        converted_asoc_prefix = self.convert_prefix(
            candidates=positive + negative_asoc,
            prompt=self.asoc_prompt_id,
            mapper=self.asoc_dict,
            ordered_prompt=self.ordered_prompt,
        )
        return converted_asoc_prefix, negative_asoc

    def full_spot(self, candidates=None, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True

        prefix_cands = candidates if candidates is not None else self.spot_list

        return self.convert_prefix(
            candidates=prefix_cands,
            prompt=self.spot_prompt_id,
            mapper=self.spot_dict,
            ordered_prompt=ordered_prompt,
        )

    def full_asoc(self, candidates=None, shuffle=False):
        # Random Prompt + Shuffle
        if not self.ordered_prompt and shuffle:
            ordered_prompt = False
        else:
            ordered_prompt = True

        prefix_cands = candidates if candidates is not None else self.asoc_list

        return self.convert_prefix(
            candidates=prefix_cands,
            prompt=self.asoc_prompt_id,
            mapper=self.asoc_dict,
            ordered_prompt=ordered_prompt,
        )

    @staticmethod
    def convert_prefix(candidates, prompt, mapper, ordered_prompt=True):
        prefix = list()

        if ordered_prompt:
            candidate_sorted = sorted([
                (candidate, index) for index, candidate in enumerate(candidates)
            ])
            index_list = [index for _, index in candidate_sorted]
        else:
            index_list = np.random.permutation(len(candidates)).tolist()

        for index in index_list:
            prefix += [prompt]
            prefix += mapper[candidates[index]]
        return prefix

class UIEDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer,
                 ssi_generator: DynamicSSIGenerator,
                 model=None,
                 label_pad_token_id=-100,
                 padding=True,
                 max_source_length: Optional[int] = None,
                 max_target_length: Optional[int] = None,
                 max_prefix_length: Optional[int] = None,
                 spot_asoc_nosier: SpotAsocNoiser = None,
                 return_tensors=True):
        self.data = data
        self.tokenizer = tokenizer
        self.ssi_generator = ssi_generator
        self.model = model
        self.label_pad_token_id = label_pad_token_id
        self.padding = padding
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.max_prefix_length = max_prefix_length
        self.spot_asoc_nosier = spot_asoc_nosier
        self.return_tensors = return_tensors
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        ins = self.data[idx]
        target_spot_asoc = copy.deepcopy(ins['spot_asoc'])
        if ins['sample_ssi'] is True:
            # Sample Dynamic SSI
            spot_prefix, pos_spot, neg_spot = self.ssi_generator.sample_spot(
                positive=ins.get('spots', []))
            asoc_prefix, neg_asoc = self.ssi_generator.sample_asoc(
                positive=ins.get('asocs', []))

            # Filter spot-asoc not in Positive Spot
            target_spot_asoc = list(
                filter(lambda x: x['label'] in pos_spot, target_spot_asoc))

            # Inject rejection noise
            if self.spot_asoc_nosier is not None:
                target_spot_asoc = self.spot_asoc_nosier.add_noise(
                    target_spot_asoc,
                    spot_label_list=neg_spot,
                    asoc_label_list=neg_asoc,
                )
        else:
            # Evaluation using Ordered SSI
            spot_prefix = self.ssi_generator.full_spot(
                shuffle=self.model.training)
            asoc_prefix = self.ssi_generator.full_asoc(
                shuffle=self.model.training)
        new_data = []
        batch = {}
        # Prepare prefix ids
        prefix = spot_prefix + asoc_prefix
        # truncate `prefix` to max length
        if self.max_prefix_length is not None and self.max_prefix_length >= 0:
            prefix = prefix[:self.max_prefix_length]
        prefix = prefix + [self.ssi_generator.text_start_id]

        # Prepare source text ids
        source_text_id = prefix + ins['input_ids']
        # truncate `input_ids` to max source length
        if self.max_source_length is not None:
            source_text_id = source_text_id[:self.max_source_length]

        # Prepare target record ids
        # Generate new record
        res = self.tokenizer.convert_ids_to_tokens(source_text_id)
        res = ''.join(res)
        target_record = convert_spot_asoc(
            target_spot_asoc, structure_maker=BaseStructureMarker())
        target_labels = self.tokenizer.encode_plus(
            target_record,
            return_token_type_ids=False,
            return_attention_mask=True,
            max_length=self.max_target_length,
            truncation=True)
        new_data.append({
                'input_ids':
                source_text_id,
                'labels':
                target_labels['input_ids'],
                'attention_mask': [1] * len(source_text_id),
                'decoder_attention_mask':
                target_labels['attention_mask'],
            })
        len_label = len(target_labels['input_ids'])
        len_source = len(source_text_id)
        def _pad_function(sequence, pad_value,k):
            if 'label' in k or 'decoder_attention_mask' in k:
                return F.pad(sequence[0],pad=(0,self.max_target_length - len_label),mode="constant",value=pad_value)
            else:
                return F.pad(sequence[0],pad=(0,self.max_source_length - len_source),mode="constant",value=pad_value)

        pad_value_map = {
            'token_type_ids': self.tokenizer.pad_token_type_id,
            'attention_mask': 0,
            'decoder_attention_mask': 0,
            'special_tokens_mask': 1,
            'input_ids': self.tokenizer.pad_token_id,
        }
        first = new_data[0]
        for k, v in first.items():
            if k not in ("labels", "label_ids"
                         ) and v is not None and not isinstance(v, str):
                # batch[k] = _pad_function(
                #     sequence=[d[k] for d in new_data],
                #     pad_value=pad_value_map[k],
                # )
                # for d in new_data:
                    # print(d[k])
                batch[k] = _pad_function(
                    sequence=[torch.tensor(d[k]) for d in new_data],
                    pad_value=pad_value_map[k],
                    k = k
                )
            else:
                batch[k] = _pad_function(
                    sequence=[torch.tensor(d[k]) for d in new_data],
                    pad_value=self.label_pad_token_id,
                    k = k
                )
        return batch