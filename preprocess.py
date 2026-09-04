import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("dataset/spam2.csv", encoding="latin1")

# Rename columns
df.columns = ["label", "message"]

# Convert labels to numbers
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Check dataset
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())

# Split dataset
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining:", len(train_df))
print("Testing :", len(test_df))