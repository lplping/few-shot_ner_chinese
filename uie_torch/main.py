#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Create by li
# Create on 2022/6/9


import argparse
from utils import *
import os
from tqdm import tqdm
from model import UIE
from tensorboardX import SummaryWriter
from data_set import UieDataSet,uie_collate_func
from mertic import SpanEvaluator
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import logging
import torch
import random
import numpy as np
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
class UieExtract(object):
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
        self.model = UIE.from_pretrained(config.model_path)
        self.model.to(self.device)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_params = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_params, lr=self.config.learning_rate)
        self.optimizer.zero_grad()

        self.train_set = None
        self.dev_set = None
        self.num_training_steps = None
        self.num_warm_steps = None
        self.scheduler = None
    def evaluate_val(self,metric):
        self.model.eval()
        self.dev_set = UieDataSet(self.config,self.config.dev_path)
        loader = DataLoader(self.dev_set, batch_size=self.config.train_batch_size,
                            shuffle=False, num_workers=4, collate_fn=uie_collate_func)
        metric.reset()

        with torch.no_grad():
            for idx, batch_data in enumerate(loader):
                input_ids = batch_data["input_ids"].to(device)
                attention_masks = batch_data["attention_masks"].to(device)
                segment_ids = batch_data["segment_ids"].to(device)
                start_ids = batch_data["start_ids"].to(device)
                end_ids = batch_data["end_ids"].to(device)
                output = self.model(input_ids, attention_masks, segment_ids,start_ids,end_ids)
                loss=output[0]
                start_prob=output[1].cpu().numpy()
                end_prob=output[2].cpu().numpy()
                num_correct, num_infer, num_label = metric.compute(start_prob, end_prob,
                                                                   start_ids, end_ids)
                metric.update(num_correct, num_infer, num_label)
            precision, recall, f1 = metric.accumulate()
            return precision, recall, f1
    def train(self):
        self.train_set = UieDataSet(self.config,self.config.train_path)
        loader = DataLoader(self.train_set, batch_size=self.config.train_batch_size,
                            shuffle=True, num_workers=4, collate_fn=uie_collate_func)

        self.num_training_steps = self.config.train_epochs * (len(self.train_set) // self.config.train_batch_size)
        self.num_warm_steps = self.config.warm_ratio* self.num_training_steps
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.num_warm_steps, num_training_steps=self.num_training_steps
        )

        global_step = 0
        best_auc = 0.0
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        writer = SummaryWriter(self.config.output_dir)
        self.model.train()
        epochs = [i for i in range(self.config.train_epochs)]
        for epoch in tqdm(epochs[0:]):

            train_batch = 0.0
            train_loss = 0.0
            for step, batch_data in enumerate(loader):
                global_step += 1

                input_ids = batch_data["input_ids"].to(device)
                attention_masks = batch_data["attention_masks"].to(device)
                segment_ids = batch_data["segment_ids"].to(device)
                # print('attention_masks',attention_masks)
                start_ids = batch_data["start_ids"].to(device)
                end_ids = batch_data["end_ids"].to(device)

                bsz = np.shape(batch_data["input_ids"]) [0]
                output = self.model(input_ids, attention_masks, segment_ids, start_ids=start_ids,end_ids=end_ids)
                loss=output[0]
                if self.config.accum_steps> 1:
                    loss = loss / self.config['accum_steps']

                train_loss += loss.item() * bsz
                train_batch += bsz

                if global_step % self.config.print_steps == 0:
                    logger.info(f'current loss is {round(train_loss/train_batch, 6)}'
                                f'at {global_step} step on {epoch} epoch...')
                    writer.add_scalar('train_loss', train_loss/train_batch, global_step)
                    train_batch = 0.0
                    train_loss = 0.0



                if global_step % self.config.eval_steps== 0:

                    metric = SpanEvaluator()
                    precision, recall, f1 = self.evaluate_val(metric)
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" %
                                (precision, recall, f1))
                    # writer.add_scalar('acc_avg', acc_avg, global_step)
                    logger.info("Evaluation precision: %.5f, recall: %.5f, F1: %.5f"
                            % (precision, recall, f1))
                    if f1 > best_auc:
                        logger.info(f'from {best_auc} -> {f1}')
                        logger.info('saving models...')
                        torch.save(self.model.state_dict(), os.path.join(self.config.output_dir, 'pytorch_model.bin'))
                        best_auc = f1

                loss.backward()

                if global_step % self.config.accum_steps== 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinidtic = True
    random.seed(seed)
    np.random.seed(seed)
def set_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='')
    parser.add_argument('--train_epochs', default=50, type=int, help='')
    parser.add_argument('--train_batch_size', default=4, type=int, help='')
    parser.add_argument('--warm_ratio', default=0.1, type=float, help='')
    parser.add_argument('--weight_decay', default=0.00, type=float)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='')
    parser.add_argument('--print_steps', default=10, type=int, help='')
    parser.add_argument('--eval_steps', default=10, type=int, help='')

    parser.add_argument('--accum_steps', default=1, type=int, help='梯度累积')
    parser.add_argument('--max_grad', default=1, type=int, help='max_grad')
    parser.add_argument("--train_path", default='data/train.json', type=str,
                        help="The path of train set.")
    parser.add_argument("--dev_path", default='data/train.json', type=str,
                        help="The path of train set.")
    parser.add_argument('--output_dir', default='model_path', type=str, help='')
    parser.add_argument('--model_path', default="E:/pre_model/bert_base_chinese_torch/",
                        type=str, help='')
    parser.add_argument('--vocab_file', default='E:/pre_model/bert_base_chinese_torch/vocab.txt', type=str, help='')

    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--max_len', type=int, default=512, help='')

    return parser.parse_args()
if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    obj=UieExtract(args)
    obj.train()
    metric = SpanEvaluator()
