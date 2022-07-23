## tool
import pickle
import tqdm
import os
import argparse
import logging
import pandas as pd
from Crawler import Crawler


def main(args):
    current_path = os.getcwd()
    logging.info(f"current python path {current_path}...")

    logging.info("Load data...")
    train_csv = pd.read_csv(f"{current_path}/data/tbrain_train_final_0603.csv")

    logging.info("Data preprocessing...")
    train_csv = pd.concat(
        [train_csv, pd.DataFrame(["dummy" for i in range(5023)])], axis=1
    )
    train_csv.columns = list(train_csv.columns[:-1]) + ["article"]
    train_csv["domain"] = train_csv["hyperlink"].apply(
        lambda x: x.split("//")[1].split("/")[0]
    )

    logging.info("Start crawling...")
    crawler = Crawler()
    crawler.crawling_process(train_csv, test=False)

    logging.info(f"Saving to {args.save_path}...")
    train_csv.to_csv(f"{args.save_path}", index=False)

    logging.info("Finish!")


def _parse_args():
    parser = argparse.ArgumentParser(description="Blacklist Prediction Preprocess")
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"{os.getcwd()}/data/train.csv",
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
