## tool
import pickle
import os
import argparse
import logging
import pandas as pd
from opencc import OpenCC

## Bert
from transformers import BertTokenizer

## module
from base.dataset import Bert_dataset

## torch
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def main(args):
    current_path = os.getcwd()
    logging.info(f"current python path {current_path}...")

    logging.info(f"Load Bert tokenizer...")
    pretrained_bert = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrained_bert, do_lower_case=True)

    cc = OpenCC("s2t")  # convert from Simplified Chinese to Traditional Chinese
    cc.set_conversion("s2tw")  # can also set conversion by calling set_conversion

    logging.info(f"Load training data...")
    with open(f"{current_path}/model_1/pkl/NER/boston.pkl", "rb") as f:
        train = pickle.load(f)

    with open(f"{current_path}/model_1/pkl/NER/asia_institute.pkl", "rb") as f:
        train2 = pickle.load(f)

    with open(f"{current_path}/model_1/pkl/NER/people.pkl", "rb") as f:
        train3 = pickle.load(f)

    all_train = train + train2 + train3

    bert_dataset = Bert_dataset(all_train)

    logging.info(f"Split dataset and save it as pkl to {args.save_path}...")
    torch.manual_seed(0)
    train_dataset, test_dataset = random_split(
        bert_dataset,
        [
            int(len(bert_dataset) * 0.8),
            len(bert_dataset) - int(len(bert_dataset) * 0.8),
        ],
    )

    valid_dataset, test_dataset = random_split(
        test_dataset,
        [
            int(len(test_dataset) * 0.5),
            len(test_dataset) - int(len(test_dataset) * 0.5),
        ],
    )

    with open(f"{args.save_path}/train_dataset.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open(f"{args.save_path}/valid_dataset.pkl", "wb") as f:
        pickle.dump(valid_dataset, f)
    with open(f"{args.save_path}/test_dataset.pkl", "wb") as f:
        pickle.dump(test_dataset, f)

    logging.info("Finish!")


def _parse_args():
    parser = argparse.ArgumentParser(description="NER Model Preprocess")
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"{os.getcwd()}/model_1/pkl/NER",
        help="the path to save dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    loglevel = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=loglevel,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args)
