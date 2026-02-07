# for data manipulation
import pandas as pd
import os

# for data preprocessing
from sklearn.model_selection import train_test_split

# for hugging face authentication & upload
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Dataset path from Hugging Face
DATASET_PATH = "hf://datasets/abhishek1504/wellness-tourism-dataset/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unique identifier
df.drop(columns=['CustomerID'], inplace=True)

# Target column
target_col = "ProdTaken"

# Split into features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test split completed and files saved locally.")

# Files to upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

# Upload to Hugging Face dataset repo
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="abhishek1504/wellness-tourism-dataset",
        repo_type="dataset",
    )

print("Files uploaded to Hugging Face successfully.")
