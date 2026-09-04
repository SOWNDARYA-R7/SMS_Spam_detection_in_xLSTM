from dataset import load_data
from model import SpamClassifier
import torch

train_df, test_df, vocab = load_data()

model = SpamClassifier(len(vocab.word2idx))

x = torch.randint(
    0,
    len(vocab.word2idx),
    (2, 50)
)

y = model(x)

print(y.shape)