from datasets import load_dataset
import sys

print("Downloading Hacker News dataset...")
try:
    # Download and cache the dataset
    ds = load_dataset("julien040/hacker-news-posts", split="train", cache_dir="./data")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    sys.exit(1)