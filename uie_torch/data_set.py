#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Create by li
# Create on 2022/5/30
"""
    文件说明:
"""
import torch
import json
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)

    return np.array(outputs)

class UieDataSet(Dataset):
    def __init__(self, args,train_path ):

        # self.tokenizer =tokenization.FullTokenizer(
        # vocab_file=args.vocab_file, do_lower_case=True)
        self.max_len = args.max_len
        self.path_files=train_path
        self.data_set = self.get_data()

    def get_data(self):

        new_data=[]
        with open(self.path_files,'r',encoding='utf-8') as f:
            for line in f:
                new_data.append(json.loads(line))


        return new_data



    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance
def uie_collate_func(batch_data):
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}
    input_ids_list, attention_masks_list,batch_segment, batch_start,batch_end = [], [], [], [],[]
    tokens_list=[]
    batch_mapping,batch_text=[],[]
    for instance in batch_data:
        token_ids = instance["input_ids"]
        attention_masks_temp = instance["attention_masks"]
        start_ids = instance["start_ids"]
        end_ids = instance["end_ids"]
        segment_ids = instance["segment_ids"]
        input_ids_list.append(token_ids)
        batch_start.append(start_ids)
        batch_end.append(end_ids)
        attention_masks_list.append(attention_masks_temp)
        batch_segment.append(segment_ids)
        tokens_list.append(instance["tokens"])
        batch_text.append(instance['text'])



    return {"input_ids": torch.tensor(sequence_padding(input_ids_list),dtype=torch.long),
            "attention_masks": torch.tensor(sequence_padding(attention_masks_list),dtype=torch.long),
            "segment_ids": torch.tensor(sequence_padding(batch_segment),dtype=torch.long),
            "start_ids": torch.tensor(sequence_padding(batch_start)),
            "end_ids": torch.tensor(sequence_padding(batch_end)),
            "tokens": tokens_list,
            'mapping':batch_mapping,
            'text':batch_text}


