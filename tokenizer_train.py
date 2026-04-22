import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# ── Step 1: Download a small Q&A dataset ──────────────────────
print("Downloading dataset...")
dataset = load_dataset("squad", split="train")

# ── Step 2: Save raw text to a file ───────────────────────────
print("Extracting text...")
os.makedirs("data", exist_ok=True)

with open("data/raw_text.txt", "w", encoding="utf-8") as f:
    for item in dataset:
        # write the context paragraph
        f.write(item["context"] + "\n")
        # write the question
        f.write(item["question"] + "\n")
        # write the answer
        f.write(item["answers"]["text"][0] + "\n")

print(f"Text saved to data/raw_text.txt")

# ── Step 3: Train the tokenizer ───────────────────────────────
print("Training tokenizer...")
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["data/raw_text.txt"],
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
)

# ── Step 4: Save the tokenizer ────────────────────────────────
os.makedirs("models/tokenizer", exist_ok=True)
tokenizer.save_model("models/tokenizer")
print("Tokenizer saved to models/tokenizer/")

# ── Step 5: Quick test ────────────────────────────────────────
encoded = tokenizer.encode("What is the capital of France?")
print(f"\nTest encode: 'What is the capital of France?'")
print(f"Tokens: {encoded.tokens}")
print(f"IDs:    {encoded.ids}")
print("\ntokenizer_train.py done!")