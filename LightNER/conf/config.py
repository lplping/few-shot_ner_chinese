#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 10:15
# @Author  : drop_lp
import yaml
class MyConf(object):
    def __init__(self, config_path):
        self.config= self.get_config(config_path)
        self.seed=self.config['seed']
        self.bart_name = self.config['bart_name']
        self.dataset_name = self.config['dataset_name']
        self.device=self.config['device']
        self.num_epochs=self.config['num_epochs']
        self.batch_size=self.config['batch_size']

        self.use_prompt = self.config['use_prompt']
        self.warmup_ratio=self.config['warmup_ratio']
        self.eval_begin_epoch=self.config['eval_begin_epoch']

        self.src_seq_ratio = self.config['src_seq_ratio']
        self.tgt_max_len = self.config['tgt_max_len']
        self.num_beams=self.config['num_beams']

        self.length_penalty=self.config['length_penalty']
        self.prompt_len=self.config['prompt_len']

        self.prompt_dim = self.config['prompt_dim']
        self.freeze_plm = self.config['freeze_plm']
        self.learn_weights = self.config['learn_weights']
        self.save_path = self.config['save_path']
        self.learning_rate = self.config['learning_rate']
        self.load_path = self.config['load_path']
        self.notes = self.config['notes']



    def get_config(self,config_path):
        with open(config_path,'r',encoding='utf-8') as f:
            config=yaml.load(f,Loader=yaml.FullLoader)
        return config

