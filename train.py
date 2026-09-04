import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import load_data, SMSDataset
from model import SpamClassifier

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device("cpu")

# -----------------------------
# Hyperparameters
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# -----------------------------
# Load Dataset
# -----------------------------
train_df, test_df, vocab = load_data()

train_dataset = SMSDataset(train_df, vocab)
test_dataset = SMSDataset(test_df, vocab)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------
model = SpamClassifier(len(vocab.word2idx)).to(DEVICE)

# -----------------------------
# Model Parameters
# -----------------------------
total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Store Training History
# -----------------------------
train_losses = []
train_accuracies = []
# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == y).sum().item()

        total += y.size(0)

    accuracy = 100 * correct / total

    train_losses.append(total_loss)
    train_accuracies.append(accuracy)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {total_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )


# -----------------------------
# Inference Latency
# -----------------------------
model.eval()

x, y = next(iter(test_loader))
x = x.to(DEVICE)

start = time.perf_counter()

with torch.no_grad():
    outputs = model(x)

end = time.perf_counter()

inference_time = end - start
latency_per_sample = inference_time / x.size(0)

print("\n==============================")
print("INFERENCE PERFORMANCE")
print("==============================")

print(f"Inference time: {inference_time:.6f} seconds")
print(f"Latency per sample: {latency_per_sample:.6f} seconds")


# -----------------------------
# Evaluation
# -----------------------------
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(DEVICE)

        outputs = model(x)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(y.numpy())
        
# -----------------------------
# Metrics
# -----------------------------
test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(f"\nTest Accuracy : {test_accuracy*100:.2f}%")

print("\nClassification Report:\n")

print(classification_report(
    all_labels,
    all_predictions,
    target_names=["Ham", "Spam"]
))

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:\n")
print(cm)

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham","Spam"],
    yticklabels=["Ham","Spam"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# -----------------------------
# Training Loss Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(
    range(1,EPOCHS+1),
    train_losses,
    marker='o',
    linewidth=2,
    label="Training Loss"
)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.savefig("training_loss.png", dpi=300)
plt.show()

# -----------------------------
# Training Accuracy Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(
    range(1,EPOCHS+1),
    train_accuracies,
    marker='o',
    linewidth=2,
    label="Training Accuracy"
)

plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.legend()

plt.savefig("training_accuracy.png", dpi=300)
plt.show()

# -----------------------------
# Save Model
# -----------------------------
torch.save(
    model.state_dict(),
    "saved_model/xlstm_model.pth"
)

# -----------------------------
# Save Vocabulary
# -----------------------------
with open("saved_model/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\nModel Saved Successfully!")
print("Vocabulary Saved Successfully!")