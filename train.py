import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
from model import SLM, SLMConfig
from tqdm import tqdm
import os

# ── Load Tokenizer ─────────────────────────────────────────────
tokenizer = ByteLevelBPETokenizer(
    "models/tokenizer/vocab.json",
    "models/tokenizer/merges.txt"
)
tokenizer.enable_padding(pad_id=0, length=256)
tokenizer.enable_truncation(max_length=256)

# ── Dataset ────────────────────────────────────────────────────
class QADataset(Dataset):
    def __init__(self, split="train"):
        print(f"Loading SQuAD {split} split...")
        raw = load_dataset("squad", split=split)

        self.samples = []
        for item in raw:
            # format: "question | answer"
            text = item["question"] + " | " + item["answers"]["text"][0]
            enc  = tokenizer.encode(text)
            ids  = enc.ids  # list of ints, padded to 256
            self.samples.append(ids)

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        # input  = all tokens except last
        # target = all tokens shifted by 1 (next token prediction)
        return ids[:-1], ids[1:]

# ── Training Loop ──────────────────────────────────────────────
def train():
    cfg     = SLMConfig()
    model   = SLM(cfg)
    device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model   = model.to(device)

    print(f"Training on: {device}")
    print(f"Model parameters: {model.count_params():,}")

    dataset    = QADataset("train")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion  = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding

    # lr scheduler — reduces lr when training plateaus
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=10
    )

    os.makedirs("models", exist_ok=True)
    best_loss = float("inf")

    print("\nStarting training...\n")

    for epoch in range(10):
        model.train()
        total_loss = 0
        batches    = 0

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")

        for input_ids, target_ids in loop:
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            # forward pass
            logits = model(input_ids)  # (B, T, vocab_size)

            # reshape for loss calculation
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                target_ids.reshape(B * T)
            )

            # backward pass
            loss.backward()

            # gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            batches    += 1

            # update progress bar
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / batches
        scheduler.step()

        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        # save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/slm_best.pt")
            print(f"  ✓ New best model saved (loss: {best_loss:.4f})")

    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print("Model saved to models/slm_best.pt")

if __name__ == "__main__":
    train()