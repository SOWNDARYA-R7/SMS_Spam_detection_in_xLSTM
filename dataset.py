import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import re

MAX_LEN = 50


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        counter = Counter()

        for text in texts:
            counter.update(clean_text(text).split())

        for word in counter:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text):
        words = clean_text(text).split()

        ids = []

        for word in words:
            ids.append(self.word2idx.get(word, 1))

        if len(ids) < MAX_LEN:
            ids += [0] * (MAX_LEN - len(ids))
        else:
            ids = ids[:MAX_LEN]

        return ids


class SMSDataset(Dataset):

    def __init__(self, dataframe, vocab):
        self.data = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        x = self.vocab.encode(row["message"])

        y = row["label"]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def load_data():

    df = pd.read_csv("dataset/spam2.csv")

    df.columns = ["label", "message"]

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    vocab = Vocabulary()
    vocab.build(train_df["message"])

    return train_df, test_df, vocab
