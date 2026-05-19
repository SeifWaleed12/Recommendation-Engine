import sys
import os
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from app.ml.retrieval.sasrec_model import SASRec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SASRecDataset(Dataset):
    def __init__(self, user_seqs, item_count, max_seq_len):
        self.user_seqs = user_seqs
        self.item_count = item_count
        self.max_seq_len = max_seq_len
        self.samples = []
        
        for uid, seq in user_seqs.items():
            if len(seq) < 2: continue
            for i in range(1, len(seq)):
                # Sequence: seq[:i], Target: seq[i]
                # Pad/Truncate sequence to max_seq_len
                sub_seq = seq[:i]
                if len(sub_seq) > max_seq_len:
                    sub_seq = sub_seq[-max_seq_len:]
                else:
                    sub_seq = [0] * (max_seq_len - len(sub_seq)) + sub_seq
                
                self.samples.append((sub_seq, seq[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, pos_item = self.samples[idx]
        
        # Negative sampling
        neg_item = random.randint(1, self.item_count)
        while neg_item == pos_item:
            neg_item = random.randint(1, self.item_count)
            
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )

def train_sasrec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / "data" / "processed"
    model_dir = project_root / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Data ────────────────────────────────────────────────────
    logger.info("Loading train interactions...")
    df = pd.read_parquet(
        data_dir / "train_interactions.parquet",
        columns=["user_ext_id", "item_ext_id", "timestamp"],
        engine="pyarrow"
    )
    
    with open(data_dir / "item_idx_map.json", "r") as f:
        item_idx_map = json.load(f)
    
    item_count = len(item_idx_map)
    
    # Sort and group
    df = df.sort_values(["user_ext_id", "timestamp"])
    
    # Map item IDs to indices (1-based, 0 is padding)
    df["item_idx"] = df["item_ext_id"].map(lambda x: item_idx_map.get(str(x), 0) + 1)
    df = df[df["item_idx"] > 0] # Remove unknown items
    
    user_seqs = df.groupby("user_ext_id")["item_idx"].apply(list).to_dict()
    
    # ── Dataset & Model ──────────────────────────────────────────────
    max_seq_len = 50
    hidden_units = 128
    
    dataset = SASRecDataset(user_seqs, item_count, max_seq_len)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    model = SASRec(
        item_count=item_count,
        max_seq_len=max_seq_len,
        hidden_units=hidden_units
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()

    # ── Training Loop ────────────────────────────────────────────────
    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for seqs, pos_items, neg_items in pbar:
            seqs, pos_items, neg_items = seqs.to(device), pos_items.to(device), neg_items.to(device)
            
            optimizer.zero_grad()
            
            user_emb = model(seqs) # (batch, hidden_units)
            
            # Scores for positive and negative items
            pos_logits = (user_emb * model.item_emb(pos_items)).sum(dim=-1)
            neg_logits = (user_emb * model.item_emb(neg_items)).sum(dim=-1)
            
            # Targets: 1 for pos, 0 for neg
            loss = criterion(pos_logits, torch.ones_like(pos_logits)) + \
                   criterion(neg_logits, torch.zeros_like(neg_logits))
            
            if not torch.isfinite(loss):
                logger.error(f"Non-finite SASRec loss ({loss.item()}); skipping batch")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, pos_items, neg_items in val_loader:
                seqs, pos_items, neg_items = seqs.to(device), pos_items.to(device), neg_items.to(device)
                user_emb = model(seqs)
                pos_logits = (user_emb * model.item_emb(pos_items)).sum(dim=-1)
                neg_logits = (user_emb * model.item_emb(neg_items)).sum(dim=-1)
                loss = criterion(pos_logits, torch.ones_like(pos_logits)) + \
                       criterion(neg_logits, torch.zeros_like(neg_logits))
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_dir / "sasrec_retriever.pt")
            logger.info("Saved best model.")

    logger.info("SASRec Training complete.")

if __name__ == "__main__":
    train_sasrec()
