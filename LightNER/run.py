import os
import logging
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from torch.utils.data import DataLoader
from deepke.name_entity_re.few_shot.models.model import PromptBartModel, PromptGeneratorModel
from deepke.name_entity_re.few_shot.module.datasets import ConllNERProcessor, ConllNERDataset
from deepke.name_entity_re.few_shot.module.train import Trainer
from deepke.name_entity_re.few_shot.module.metrics import Seq2SeqSpanMetric
from deepke.name_entity_re.few_shot.utils.util import get_loss, set_seed
from deepke.name_entity_re.few_shot.module.mapping_type import mit_movie_mapping, mit_restaurant_mapping, atis_mapping
from conf.config import MyConf
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
'''
基于fnlp/bart-large-chinese
'''

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


DATASET_CLASS = {
    'conll2003': ConllNERDataset,
    'mit-movie': ConllNERDataset,
    'daily_people': ConllNERDataset,
    'mit-restaurant': ConllNERDataset,
    'atis': ConllNERDataset
}

DATA_PROCESS = {
    'conll2003': ConllNERProcessor,
    'mit-movie': ConllNERProcessor,
    'daily_people': ConllNERProcessor,
    'mit-restaurant': ConllNERProcessor,
    'atis': ConllNERProcessor
}

DATA_PATH = {
    'daily_people': {'train': 'data/daily_people/5-shot-train.txt',
                  'dev': 'data/daily_people/test.txt'},
    'mit-restaurant': {'train': 'data/mit-restaurant/10-shot-train.txt',
                  'dev': 'data/mit-restaurant/test.txt'},
    'atis': {'train': 'data/atis/20-shot-train.txt',
                  'dev': 'data/atis/test.txt'}
}

MAPPING = {
    'conll2003': {'loc': '<<location>>',
                'per': '<<person>>',
                'org': '<<organization>>',
                'misc': '<<others>>'},
    'daily_people': {'loc': '<<location>>',
                'per': '<<person>>',
                'org': '<<organization>>',
                'pro': '<<product>>'},
    'mit-movie': mit_movie_mapping,
    'mit-restaurant': mit_restaurant_mapping,
    'atis': atis_mapping
}

def main():

    cfg=MyConf('conf/few_shot.yaml')


    device = torch.device("cuda" if torch.cuda.is_available() and int(cfg.device) >= 0 else "cpu")

    
    data_path = DATA_PATH[cfg.dataset_name]
    dataset_class, data_process = DATASET_CLASS[cfg.dataset_name], DATA_PROCESS[cfg.dataset_name]
    mapping = MAPPING[cfg.dataset_name]

    set_seed(cfg.seed) # set seed, default is 1
    if cfg.save_path is not None:  # make save_path dir
        cfg.save_path = os.path.join(cfg.save_path, cfg.dataset_name+"_"+str(cfg.batch_size)+"_"+str(cfg.learning_rate)+cfg.notes)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path, exist_ok=True)
    
    process = data_process(data_path=data_path, mapping=mapping, bart_name=cfg.bart_name, learn_weights=cfg.learn_weights)
    train_dataset = dataset_class(data_processor=process, mode='train')
    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.collate_fn, batch_size=cfg.batch_size, num_workers=4)
    
    dev_dataset = dataset_class(data_processor=process, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, collate_fn=dev_dataset.collate_fn, batch_size=cfg.batch_size, num_workers=4)
    label_ids = list(process.mapping2id.values())# [[12264, 11424], [9205, 8640, 8224], [8448, 13154, 9283, 8361], [11691]]

    prompt_model = PromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=cfg,device=device)
    model = PromptGeneratorModel(prompt_model=prompt_model, bos_token_id=0,
                                eos_token_id=1,
                                max_length=cfg.tgt_max_len, max_len_a=cfg.src_seq_ratio,num_beams=cfg.num_beams, do_sample=False,
                                repetition_penalty=1, length_penalty=cfg.length_penalty, pad_token_id=0,
                                restricter=None)
    metrics = Seq2SeqSpanMetric(eos_token_id=1, num_labels=len(label_ids), target_type='word')
    loss = get_loss

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=None, model=model, process=process,args=cfg, logger=logger, loss=loss,
                      metrics=metrics, device=device)
    trainer.train()



if __name__ == "__main__":
    main()
