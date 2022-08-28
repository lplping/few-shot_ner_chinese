#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Create by li
# Create on 2022/6/13
import numpy as np
from collections import defaultdict
def get_bool_ids_greater_than(probs, limit=0.5, return_prob=False):
    """
    Get idx of the last dimension in probability arrays, which is greater than a limitation.

    Args:
        probs (List[List[float]]): The input probability arrays.
        limit (float): The limitation for probability.
        return_prob (bool): Whether to return the probability
    Returns:
        List[List[int]]: The index of the last dimension meet the conditions.
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result
def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.

    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once .
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue
        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue
        if start_id > end_id:
            end_pointer += 1
            continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def get_id_and_prob(spans,unk_tokens,text,mapping):
    '''
    对bert后的索引转化为原始索引，计算offset
    :param spans:
    :return:
    '''
    sentence_id = []
    prob = []
    sep_index=unk_tokens.index('[SEP]')

    for start, end in spans:
        prompt_len=sep_index
        prob.append(start[1] * end[1])
        start_new=start[0]-prompt_len
        end_new=end[0]-prompt_len

        t_to = ''.join(unk_tokens[start[0]:end[0] + 1])
        t_to = t_to.replace('##', '')
        # print(t_to)
        # print('text',text[mapping[start_new][0]: mapping[end_new][-1]+1].lower())
        assert t_to==text[mapping[start_new][0]: mapping[end_new][-1]+1].lower()
        sentence_id.append((mapping[start_new][0], mapping[end_new][-1]))

    return sentence_id, prob

