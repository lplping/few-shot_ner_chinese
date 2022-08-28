import torch
from tqdm import tqdm
import numpy as np
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer,BertTokenizer

import logging
logger = logging.getLogger(__name__)


# load file and process bio
class ConllNERProcessor(object):
    def __init__(self, data_path, mapping, bart_name, learn_weights) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bart_name)
        self.tokenizer.bos_token = "[CLS]"
        self.tokenizer.eos_token = "[SEP]"
        # self.tokenizer = BartTokenizer.from_pretrained(bart_name)
        self.mapping = mapping  # 记录的是原始tag与转换后的tag的str的匹配关系
        self.original_token_nums = self.tokenizer.vocab_size
        self.learn_weights = learn_weights
        self._add_tags_to_tokens()

    def load_from_file(self, mode='train'):
        """load conll ner from file

        Args:
            mode (str, optional): train/test/dev. Defaults to 'train'.
        Return:
            outputs (dict)
            raw_words: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
            raw_targets: ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']
            entities: [['EU'], ['German'], ['British']]
            entity_tags: ['org', 'misc', 'misc']
            entity_spans: [[0, 1], [2, 3], [6, 7]]
        """
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))

        # extract bio
        split_c = '\t' if 'conll' in load_file  else ' '
        outputs = {'raw_words':[], 'raw_targets':[], 'entities':[], 'entity_tags':[], 'entity_spans':[],'old_labels':[]}
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            for line in lines:
                if line != "\n":
                    raw_word.append(line.split(split_c)[0])
                    raw_target.append(line.split(split_c)[1][:-1])
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        for words, targets in zip(raw_words, raw_targets):
            entities, entity_tags, entity_spans = [], [], []
            old_label=[]
            start, end, start_flag = 0, 0, False
            for idx, tag in enumerate(targets):
                old_label.append(tag)
                if tag.startswith('B-'):    # 一个实体开头 另一个实体（I-）结束
                    end = idx
                    if start_flag:  # 另一个实体以I-结束，紧接着当前实体B-出现
                        entities.append(words[start:end])
                        entity_tags.append(targets[start][2:].lower())
                        entity_spans.append([start, end])
                        start_flag = False
                    start = idx
                    start_flag = True
                elif tag.startswith('I-'):  # 实体中间，不是开头也不是结束，end+1即可
                    end = idx
                elif tag.startswith('O'):  # 无实体，可能是上一个实体的结束
                    end = idx
                    if start_flag:  # 上一个实体结束
                        entities.append(words[start:end])
                        entity_tags.append(targets[start][2:].lower())
                        entity_spans.append([start, end])
                        start_flag = False
            if start_flag:  # 句子以实体I-结束，未被添加
                entities.append(words[start:end+1])
                entity_tags.append(targets[start][2:].lower())
                entity_spans.append([start, end+1])
                start_flag = False
    
            if len(entities) != 0:
                outputs['raw_words'].append(words)
                outputs['raw_targets'].append(targets)
                outputs['entities'].append(entities)
                outputs['entity_tags'].append(entity_tags)
                outputs['entity_spans'].append(entity_spans)
                outputs['old_labels'].append(old_label)
        return outputs

    def process(self, data_dict):
        target_shift = len(self.mapping) + 2
        # print('self.mapping',self.mapping)
        def prepare_target(item):
            # print('self.mapping2targetid',self.mapping2targetid)
            raw_word = item['raw_word']
            # print('raw_word',raw_word)
            word_bpes = [[self.tokenizer.bos_token_id]] 
            first = []  #记录被token后的raw_word长度
            cur_bpe_len = 1
            for word in raw_word:
                # print('word',word)
                bpes = self.tokenizer.tokenize(word)
                # print('bpes',bpes)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                first.append(cur_bpe_len)
                cur_bpe_len += len(bpes)
                word_bpes.append(bpes)
            assert first[-1] + len(bpes) == sum(map(len, word_bpes))
            word_bpes.append([self.tokenizer.eos_token_id])
            #word_bpes [[0], [11], [4525], [449, 225, 2636], [7458, 1688], [222], [10], [1569], [14], [56], [10], [691], [9], [411], [99], [21], [24], [373], [2]]
            #first [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            # print('word_bpes',word_bpes)
            assert len(first) == len(raw_word) == len(word_bpes) - 2

            lens = list(map(len, word_bpes)) #lens [1, 1, 1, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            #lens记录每个字符被token成几个字符
            # print('first',first)
            # print('lens',lens)
            cum_lens = np.cumsum(lens).tolist()
            #cum_lens [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
            # print('cum_lens',cum_lens)

            entity_spans = item['entity_span']  # [(s1, e1, s2, e2), ()]
            entity_tags = item['entity_tag']  # [tag1, tag2...]
            entities = item['entity']  # [[ent1, ent2,], [ent1, ent2]]
            target = [0]
            pairs = []

            first = list(range(cum_lens[-1]))#[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
            # print('first2', first)

            assert len(entity_spans) == len(entity_tags)                #
            # print('entity_spans',entity_spans)
            # print('entity_tags',entity_tags)
            for idx, (entity, tag) in enumerate(zip(entity_spans, entity_tags)):
                cur_pair = []
                num_ent = len(entity) // 2
                # print('num_ent',num_ent)
                # print('entity',entity)
                # print('tag',tag)
                for i in range(num_ent):
                    # print('i',i)
                    start = entity[2 * i]
                    end = entity[2 * i + 1]
                    cur_pair_ = []
                    cur_pair_.extend([cum_lens[k] for k in list(range(start, end))])
                    # print('start',start)
                    # print('end', end)
                    # print('cur_pair_',cur_pair_)
                    cur_pair.extend([p + target_shift for p in cur_pair_])
                    # print('cur_pair_', cur_pair)
                # for _, (j, word_idx) in enumerate(zip((cur_pair[0], cur_pair[-1]), (0, -1))):
                #     j = j - target_shift
                assert all([cur_pair[i] < cum_lens[-1] + target_shift for i in range(len(cur_pair))])

                cur_pair.append(self.mapping2targetid[tag] + 2)
                # print('cur_pair',cur_pair)
                pairs.append([p for p in cur_pair])
            # print('pairs',pairs)
            target.extend(list(chain(*pairs)))
            target.append(1)
            # print('target',target)
            # exit()

            word_bpes = list(chain(*word_bpes))
            #[0, 11, 4525, 449, 225, 2636, 7458, 1688, 222, 10, 1569, 14, 56, 10, 691, 9, 411, 99, 21, 24, 373, 2]
            # print('word_bpes2',word_bpes)
            assert len(word_bpes)<510

            dict  = {'tgt_tokens': target, 'target_span': pairs, 'src_tokens': word_bpes,
                    'first': first, 'src_seq_len':len(word_bpes), 'tgt_seq_len':len(target)}
            return dict
        
        logger.info("Process data...")
        for raw_word, raw_target, entity, entity_tag, entity_span in tqdm(zip(data_dict['raw_words'], data_dict['raw_targets'], data_dict['entities'], 
                                                                                data_dict['entity_tags'], data_dict['entity_spans']), total=len(data_dict['raw_words']), desc='Processing'):
            item_dict = prepare_target({'raw_word': raw_word, 'raw_target':raw_target, 'entity': entity, 'entity_tag': entity_tag, 'entity_span': entity_span})
            # add item_dict to data_dict
            for key, value in item_dict.items():
                if key in data_dict:
                    data_dict[key].append(value)
                else:
                    data_dict[key] = [value]
        return data_dict

    def _add_tags_to_tokens(self):
        mapping = self.mapping
        if self.learn_weights:  # add extra tokens to huggingface tokenizer
            # print('##############3')
            self.mapping2id = {} 
            self.mapping2targetid = {} 
            for key, value in self.mapping.items():
                # print(key, value)#loc << location >>
                # print('value[2:-2]',value[2:-2])#location
                key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value[2:-2]))
                self.mapping2id[value] = key_id  # may be list
                self.mapping2targetid[key] = len(self.mapping2targetid)
        else:
            # print()
            tokens_to_add = sorted(list(mapping.values()), key=lambda x: len(x), reverse=True)  # 
            unique_no_split_tokens = self.tokenizer.unique_no_split_tokens                      # no split
            sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x: len(x), reverse=True)
            for tok in sorted_add_tokens:
                assert self.tokenizer.convert_tokens_to_ids([tok])[0] == self.tokenizer.unk_token_id    # 
            self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens          # add to no_split_tokens
            self.tokenizer.add_tokens(sorted_add_tokens)
            self.mapping2id = {}  # tag to id
            self.mapping2targetid = {}  # tag to number

            for key, value in self.mapping.items():
                key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
                assert len(key_id) == 1, value
                assert key_id[0] >= self.original_token_nums
                self.mapping2id[value] = key_id[0]  #
                self.mapping2targetid[key] = len(self.mapping2targetid)
        

class ConllNERDataset(Dataset):
    def __init__(self, data_processor, mode='train') -> None:
        self.data_processor = data_processor
        # print(' self.mode',mode)
        self.data_dict = data_processor.load_from_file(mode=mode)
        self.complet_data = data_processor.process(self.data_dict)
        self.mode = mode


    def __len__(self):
        return len(self.complet_data['src_tokens'])

    def __getitem__(self, index):
        if self.mode == 'test':
            return torch.tensor(self.complet_data['src_tokens'][index]), torch.tensor(self.complet_data['src_seq_len'][index]), \
                    torch.tensor(self.complet_data['first'][index]), self.complet_data['raw_words'][index]
        # print('src_tokens', self.complet_data['src_tokens'][0])
        # print(self.complet_data['src_tokens'][index])
        # print('tgt_tokens', self.complet_data['tgt_tokens'][0])

        # print('src_tokens', len(self.complet_data['src_tokens']))
        # # print('src_tokens_idx',self.complet_data['src_tokens'][index])
        # print('tgt_tokens', len(self.complet_data['tgt_tokens']))
        # # print('tgt_tokens_idx', self.complet_data['tgt_tokens'][index])
        # # print('src_seq_len', tup[2])
        # # print('tgt_seq_len', tup[3])
        # # print('first', tup[4])
        # # print('target_span', tup[5])
        # exit()

        return torch.tensor(self.complet_data['src_tokens'][index]), torch.tensor(self.complet_data['tgt_tokens'][index]), \
                    torch.tensor(self.complet_data['src_seq_len'][index]), torch.tensor(self.complet_data['tgt_seq_len'][index]), \
                    torch.tensor(self.complet_data['first'][index]), self.complet_data['target_span'][index],\
               self.complet_data['raw_words'][index],self.complet_data['old_labels'][index]


    def collate_fn(self, batch):
        src_tokens, src_seq_len, first  = [], [], []
        tgt_tokens, tgt_seq_len, target_span = [], [], []
        raw_words=[]
        old_labels=[]
        if self.mode == "test":
            raw_words = []
            for tup in batch:
                src_tokens.append(tup[0])
                src_seq_len.append(tup[1])
                first.append(tup[2])
                raw_words.append(tup[3])
            src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.data_processor.tokenizer.pad_token_id)
            first = pad_sequence(first, batch_first=True, padding_value=0)
            return src_tokens, torch.stack(src_seq_len, 0), first, raw_words

        for tup in batch:
            src_tokens.append(tup[0])
            tgt_tokens.append(tup[1])
            src_seq_len.append(tup[2])
            tgt_seq_len.append(tup[3])
            first.append(tup[4])
            target_span.append(tup[5])
            raw_words.append(tup[6])
            old_labels.append(tup[7])

        src_tokens = pad_sequence(src_tokens, batch_first=True, padding_value=self.data_processor.tokenizer.pad_token_id)
        tgt_tokens = pad_sequence(tgt_tokens, batch_first=True, padding_value=0)
        first = pad_sequence(first, batch_first=True, padding_value=0)
        return src_tokens, tgt_tokens, torch.stack(src_seq_len, 0), torch.stack(tgt_seq_len, 0), first, target_span,raw_words,old_labels


if __name__ == '__main__':
    data_path = {'train':'data/conll2003/train.txt'}
    bart_name = '../BARTNER-AMAX/facebook/'
    conll_processor = ConllNERProcessor(data_path, bart_name)
    conll_datasets = ConllNERDataset(conll_processor, mode='train')
    conll_dataloader = DataLoader(conll_datasets, collate_fn=conll_datasets.collate_fn, batch_size=8)
    for idx, data in enumerate(conll_dataloader):
        print(data)
        break
    
