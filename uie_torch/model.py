#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Create by li
# Create on 2022/6/10
import torch.nn as nn
from transformers import BertModel,BertPreTrainedModel



class UIE(BertPreTrainedModel):
    def __init__(self, config):

        super().__init__(config)
        self.bert = BertModel(config)
        hidden_size =self.bert.config.hidden_size
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()


    def forward(self, input_ids, attention_mask, token_type_ids,start_ids=None,end_ids=None):


        context_outputs = self.bert(input_ids, attention_mask, token_type_ids
                                    )
        sequence_output=context_outputs[0]

        start_logits = self.linear_start(sequence_output)
        start_logits = start_logits.squeeze( -1)
        start_prob = self.sigmoid(start_logits)
        end_logits = self.linear_end(sequence_output)
        end_logits = end_logits.squeeze(-1)
        end_prob = self.sigmoid(end_logits)
        out=(start_prob, end_prob)
        if start_ids!=None:
            loss_start = self.criterion(start_prob, start_ids.float())
            loss_end = self.criterion(end_prob, end_ids.float())
            loss = (loss_start + loss_end) / 2.0

            out=(loss,)+out

        return out
