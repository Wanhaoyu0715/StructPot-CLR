import logging
import math
import os
import random
import time

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import get_scheduler

from src.data.dataset import CharDataset, MyCollator
from src.models.clip_model import CLIP, CLIPConfig, PointNetConfig
from src.models.crystal_encoder import cry_config, CRY_ENCODER

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Configuration
device = 'cuda:1'
numEpochs = 1000
embeddingSize = 384
batchSize = 64
n_layers = 2
n_heads = 4
numPoints = 9
blockSize = 95
output_dir = './checkpoints'

# Load database
db = connect('./data/processed/structures.db')
rows = list(db.select())

dataset = CharDataset(rows)

# Split dataset
total_size = len(dataset)
train_size = int(total_size * 0.8)
test_size = int(total_size * 0.1)
validation_size = total_size - train_size - test_size

indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + validation_size]
test_indices = indices[train_size + validation_size:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create model
mconf = cry_config(blockSize, n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)
cry_encoder = CRY_ENCODER(mconf)

pconf = PointNetConfig(embeddingSize=embeddingSize, numberofPoints=numPoints)
mconf = CLIPConfig(blockSize, n_layer=n_layers, n_head=n_heads, n_embd=embeddingSize)
model = CLIP(mconf, pconf, cry_encoder)

# Create data loaders
collator = MyCollator(mysql_url=db)
train_loader = DataLoader(
    train_dataset, batch_size=batchSize, num_workers=8,
    collate_fn=collator, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=2*batchSize, num_workers=4,
    collate_fn=collator, shuffle=True, drop_last=False
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

# Training setup
num_update_steps_per_epoch = len(train_loader)
max_train_steps = numEpochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=max_train_steps,
)

best_loss = float('inf')

# Training loop
for epoch in range(numEpochs):
    model.train()
    t0 = time.time()

    pbar = tqdm(train_loader, total=len(train_loader))
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, logits = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        pbar.set_description(f"Epoch {epoch+1} Step {step}: loss {loss.item():.4f}, lr {lr:.2e}")

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, logits = model(batch)
            val_losses.append(loss.item())
    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch {epoch+1} validation loss: {avg_val_loss:.4f}  time: {time.time()-t0:.1f}s")

    # Save best model
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        print("Saving best model...")
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "best_contra.pt"))