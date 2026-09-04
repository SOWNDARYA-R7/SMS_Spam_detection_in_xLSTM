import torch
import pickle

from model import SpamClassifier
from dataset import clean_text, MAX_LEN

DEVICE = torch.device("cpu")

# Load vocabulary
with open("saved_model/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Load model
model = SpamClassifier(len(vocab.word2idx))
model.load_state_dict(torch.load("saved_model/xlstm_model.pth", map_location=DEVICE))
model.eval()


def encode(text):
    words = clean_text(text).split()

    ids = []

    for word in words:
        ids.append(vocab.word2idx.get(word, 1))

    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN]

    return torch.tensor([ids], dtype=torch.long)


while True:

    message = input("\nEnter SMS (or type 'exit'): ")

    if message.lower() == "exit":
        break

    x = encode(message)

    with torch.no_grad():

        output = model(x)

        probabilities = torch.softmax(output, dim=1)

        confidence, prediction = torch.max(probabilities, dim=1)

    if prediction.item() == 1:
        print(f"\nPrediction : SPAM")
    else:
        print(f"\nPrediction : HAM")

    print(f"Confidence : {confidence.item() * 100:.2f}%")