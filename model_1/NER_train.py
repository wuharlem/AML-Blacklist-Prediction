## tool
import pickle
import tqdm
import os
import argparse
import logging

## Bert
from transformers import BertForTokenClassification

## module
from module.trainer import Trainer
from module.dataset import Bert_dataset

## torch
import torch
from torch.utils.data import DataLoader

def main(args):
    current_path = os.getcwd()
    logging.info(f'current python path {current_path}...')
    logging.info('Load data...')
    
    with open(f"{args.dataset}/train_dataset.pkl", "rb") as f:
        train_dataset = pickle.load(f)
    with open(f"{args.dataset}/valid_dataset.pkl", "rb") as f:
        valid_dataset = pickle.load(f)
    with open(f"{args.dataset}/test_dataset.pkl", "rb") as f:
        test_dataset = pickle.load(f)
    
    logging.info('Making dataloader...')
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        collate_fn = lambda x: Bert_dataset.collate_fn(train_dataset, x)
    )

    valid_loader = DataLoader(
        dataset = valid_dataset,
        batch_size = args.batch_size,
        collate_fn = lambda x: Bert_dataset.collate_fn(valid_dataset, x)
    )

    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        collate_fn = lambda x: Bert_dataset.collate_fn(test_dataset, x)
    )
    
    logging.info('Load model and parameters...')
    model = BertForTokenClassification.from_pretrained("bert-base-chinese",
        num_labels = 3,
        output_attentions = False,
        output_hidden_states = False
    )
    
    trainer = Trainer(model, train_loader, valid_loader)
    
    logging.info('Test validation dataset...')
    acc, total_loss = trainer.evaluation(test=False)
    print(f"device: {trainer.device} classification acc: {acc: .4f} validation loss: {total_loss:.4f}")
    
    logging.info('Start training...')
    trainer.training_process(early_stopping = True, 
                             n_iter_no_change = 5, 
                             max_epoch = args.max_epoch, 
                             save_params = True, 
                             verbose = True, 
                             learning_rate = args.learning_rate, 
                             save_paths = args.save_paths)
    
    logging.info('Training ends!')
    logging.info('Test validation dataset...')
    acc, total_loss = trainer.evaluation(test=False)
    print(f"device: {trainer.device} classification acc: {acc: .4f} validation loss: {total_loss:.4f}")
    logging.info('Finish!')

    
def _parse_args():
    current_path = os.getcwd()
    parser = argparse.ArgumentParser(description='NER Model Training')
    parser.add_argument('--save_paths', type=str, default = f'{current_path}/model_1/params/pretrained/test.pth', 
                        help = 'the path of saved parameter')
    parser.add_argument('--dataset', type=str, default = f'{current_path}/model_1/pkl/NER', 
                        help = 'the path of dataset')
    parser.add_argument('--batch_size', type=int, default = 64, 
                        help = 'batch size')
    parser.add_argument('--max_epoch', type=int, default = 10, 
                        help = 'max number of epoch')
    parser.add_argument('--learning_rate', type=float, default = 1e-5, 
                        help = 'learning rate')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)
