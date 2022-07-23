## tool
import pickle
import tqdm
import os
import argparse
import logging
import pandas as pd

## Bert
from transformers import BertTokenizer

## module
from base.dataset import Bert_Blacklist_dataset

## torch
import torch
from torch.utils.data import random_split, Dataset


def main(args):
    current_path = os.getcwd()
    logging.info(f"current python path {current_path}...")
    logging.info("Load data...")
    train_csv = pd.read_csv(args.dataset)

    logging.info("Load bert model...")
    tokenizer_chinese = BertTokenizer.from_pretrained(
        "bert-base-chinese", do_lower_case=False
    )

    logging.info("Data Preprocessing...")
    train_df = train_csv.drop(["hyperlink", "content", "domain", "name"], axis=1)
    train_df["length"] = train_csv["article"].apply(lambda x: len(x))
    train_df["sentences"] = train_csv["article"].apply(lambda x: split_sentence(x))
    train_df["token_ids"] = train_df["sentences"].apply(
        lambda x: [tokenizer_chinese.encode(sentence) for sentence in x]
    )
    train_df["blacklist"] = train_csv["name"].apply(lambda x: str2list(x))

    ## delete some useless data
    train_df = train_df[train_df.article != "文章已被刪除 404 or 例外"]
    train_df = train_df[(train_df["length"]) <= 2551]
    train_df = train_df[(train_df["length"]) >= 75]
    train_df = train_df.reset_index(drop=True)

    logging.info("Making data collection...")
    data_collection = []

    for data in tqdm.tqdm(train_df.values):

        news_ID, article, length, sentences, token_ids, blacklist = data

        for token_id, sentence in zip(token_ids, sentences):
            sample = {}
            sample["ID"] = news_ID
            sample["original_article"] = article
            sample["length"] = length
            sample["sentence"] = sentence
            sample["token_id"] = token_id
            sample["blacklist"] = blacklist

            ## get label
            one_hot = torch.zeros(len(token_id)).tolist()
            one_hot2 = torch.zeros(len(token_id)).tolist()

            position = []

            for black_name in blacklist:
                black_name_ids = tokenizer_chinese.encode(black_name)[1:-1]
                position += get_index(black_name_ids, token_id)

            for i in position:
                one_hot[i] = 1

            sample["label"] = one_hot

            data_collection.append(sample)

    bert_dataset = Bert_Blacklist_dataset(data_collection)

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


def split_sentence(text):
    sentences = []
    if len(text) < 500:
        sentences.append(text)

    elif len(text) > 500 and len(text) <= 1000:
        mid = int(len(text) / 2)
        sentences.append(text[:500])
        sentences.append(text[mid - 250 : mid + 250])
        sentences.append(text[-500:])

    elif len(text) > 1000 and len(text) <= 1500:
        mid = int(len(text) / 2)
        sentences.append(text[:500])
        sentences.append(text[mid - 250 : mid + 250])
        sentences.append(text[-500:])

    elif len(text) > 1500 and len(text) <= 2000:
        mid = int(len(text) / 2)
        point_1 = int(len(text) * 0.25)
        point_2 = int(len(text) * 0.75)
        sentences.append(text[:500])
        sentences.append(text[point_1 - 250 : point_1 + 250])
        sentences.append(text[mid - 250 : mid + 250])
        sentences.append(text[point_2 - 250 : point_2 + 250])
        sentences.append(text[-500:])

    else:
        mid = int(len(text) / 2)
        point_1 = int(len(text) * 0.25)
        point_2 = int(len(text) * 0.75)
        sentences.append(text[:500])
        sentences.append(text[point_1 - 250 : point_1 + 250])
        sentences.append(text[mid - 250 : mid + 250])
        sentences.append(text[point_2 - 250 : point_2 + 250])
        sentences.append(text[-500:])

    return sentences


def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(seq[:to_len] + [padding] * max(0, to_len - len(seq)))
    return paddeds


def str2list(s):
    if s == "[]":
        return []
    return [str(i.replace(" ", "")[1:-1]) for i in s[1:-1].split(",")]


def get_index(name_index, sentence_index):

    if name_index == []:
        return []

    arr = []
    j = 0
    l_name = len(name_index)
    l_sentence = len(sentence_index)

    while j < l_sentence:
        if name_index[0] == sentence_index[j]:

            flag = 1
            record = j

            for i in range(l_name):
                if name_index[i] != sentence_index[j]:
                    flag = 0
                    break
                j += 1

            if flag:
                j -= 1
                arr += [i for i in range(record, record + l_name)]
        j += 1

    return arr


def _parse_args():
    parser = argparse.ArgumentParser(description="Blacklist Prediction Preprocess")
    parser.add_argument(
        "--dataset",
        type=str,
        default=f"{os.getcwd()}/data/train.csv",
        help="the path of dataset",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"{os.getcwd()}/model_1/pkl",
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
