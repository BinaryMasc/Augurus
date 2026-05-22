import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import FinancialTransformer
from config import MODEL_CONFIG, WEIGHT_DECAY, GAUSSIAN_NOISE
import os

class FinancialDataset(Dataset):
    def __init__(self, df, seq_len=128, is_training=False):
        self.seq_len = seq_len
        self.is_training = is_training
        
        # Continuous features: hour_sin, hour_cos, day_sin, day_cos, norm_return, ema12_dist, ema26_dist
        cont_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'norm_return', 'ema12_dist', 'ema26_dist']
        
        # Verify format
        missing = [c for c in cont_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataframe: {missing}")
            
        self.tokens = torch.tensor(df['token'].values, dtype=torch.long)
        self.continuous = torch.tensor(df[cont_cols].values, dtype=torch.float32)

    def __len__(self):
        # We need seq_len elements for input, plus 1 for target
        return len(self.tokens) - self.seq_len - 1

    def __getitem__(self, idx):
        # Input sequence (t to t + seq_len - 1)
        x_tok = self.tokens[idx : idx + self.seq_len]
        x_cont = self.continuous[idx : idx + self.seq_len].clone()
        
        # Adding noise to continuous features during training (prevent memorization)
        if self.is_training:
            noise = torch.randn_like(x_cont) * GAUSSIAN_NOISE
            x_cont = x_cont + noise
        
        # Target sequence (t+1 to t + seq_len)
        # The transformer predicts the next token at each step
        y_tok = self.tokens[idx + 1 : idx + self.seq_len + 1]
        
        return x_tok, x_cont, y_tok

def train_phase1():
    # 1. Configuration
    SEQ_LEN = MODEL_CONFIG['seq_len']
    BATCH_SIZE = 128
    EPOCHS = 14  
    LR = 2e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Force CPU for now
    # DEVICE = torch.device("cpu")
    #torch.set_num_threads(os.cpu_count() or 4) # Optimize CPU usage
    

    print(f"Using device: {DEVICE}")

    # 2. Load Data
    data_path = "processed/tokenized_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find {data_path}. Did you run process_data.py?")
        
    print("Loading preprocessed data...")
    df = pd.read_csv(data_path)
    
    # Split into Train/Val (90/10 chronologically)
    split_idx = int(len(df) * 0.9)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")
    
    train_dataset = FinancialDataset(train_df, seq_len=SEQ_LEN, is_training=True)
    val_dataset = FinancialDataset(val_df, seq_len=SEQ_LEN, is_training=False)
    
    # Use multiple workers if on a strong machine to speed up data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # 3. Initialize Model and load checkpoint if exists
    model = FinancialTransformer(**MODEL_CONFIG).to(DEVICE)
    
    start_epoch = 0

    # TEMP
    checkpoint_path = "processed/model_epoch_6_6.45.pt"
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
            start_epoch = 5
        except RuntimeError as e:
            print(f"Warning: Checkpoint shape mismatch detected ({e}).")
            print("Initializing model from scratch due to architecture changes.")
    
    #
    # Increased weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # Added label smoothing so the model doesn't become overconfident
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Added a learning rate scheduler to decay LR over time
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))
    
    print(f"\nStarting Phase 1 Training (Next-Token Prediction) on {DEVICE}...")
    for epoch in range(start_epoch, EPOCHS):
        # Manual learning rate decay at epoch 5 (Epoch 6)
        if epoch == 5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-5  # Set LR lower
            # Update scheduler base LRs so it continues annealing correctly from the new level
            scheduler.base_lrs = [5e-5 for _ in scheduler.base_lrs]
            print(f"\n[LR Scheduler] Manual learning rate decay at epoch {epoch+1}. New LR: {optimizer.param_groups[0]['lr']}")

        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
            
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for x_tok, x_cont, y_tok in progress_bar:
            x_tok, x_cont, y_tok = x_tok.to(DEVICE), x_cont.to(DEVICE), y_tok.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _, _ = model(x_tok, x_cont)
            
            # Flatten for CrossEntropyLoss
            # logits: [batch, seq_len, vocab_size] -> [batch * seq_len, vocab_size]
            # y_tok: [batch, seq_len] -> [batch * seq_len]
            token_loss = criterion(logits.view(-1, 256), y_tok.view(-1))
            
            # Auxiliary Directional Loss (Binary Cross Entropy)
            probs = torch.softmax(logits, dim=-1)
            pred_up = probs[..., 128:].sum(dim=-1).view(-1)
            true_up = (y_tok.view(-1) > 127).float()
            bce_loss = nn.functional.binary_cross_entropy(pred_up.clamp(1e-7, 1.0 - 1e-7), true_up)
            
            # Combine losses to directly optimize direction prediction
            loss = token_loss + 1.0 * bce_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step the scheduler each batch
            
            total_loss += loss.item()
            
            # Calculate direction accuracy
            preds = torch.argmax(logits, dim=-1)
            # Tokens > 127 are positive returns (Up), <= 127 are negative/neutral (Down)
            pred_up = preds > 127
            true_up = y_tok > 127
            total_correct += (pred_up == true_up).sum().item()
            total_samples += y_tok.numel()
            
            acc = total_correct / total_samples * 100
            progress_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
            
        avg_train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples * 100
        
        # Validation Loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for x_tok, x_cont, y_tok in val_bar:
                x_tok, x_cont, y_tok = x_tok.to(DEVICE), x_cont.to(DEVICE), y_tok.to(DEVICE)
                logits, _, _ = model(x_tok, x_cont)
                token_loss = criterion(logits.view(-1, 256), y_tok.view(-1))
                
                # Auxiliary Directional Loss (Binary Cross Entropy)
                probs = torch.softmax(logits, dim=-1)
                pred_up = probs[..., 128:].sum(dim=-1).view(-1)
                true_up = (y_tok.view(-1) > 127).float()
                bce_loss = nn.functional.binary_cross_entropy(pred_up.clamp(1e-7, 1.0 - 1e-7), true_up)
                
                loss = token_loss + 1.0 * bce_loss
                val_loss += loss.item()
                
                # Calculate direction accuracy
                preds = torch.argmax(logits, dim=-1)
                # Tokens > 127 are positive returns (Up), <= 127 are negative/neutral (Down)
                pred_up = preds > 127
                true_up = y_tok > 127
                val_correct += (pred_up == true_up).sum().item()
                val_samples += y_tok.numel()
                
                acc = val_correct / val_samples * 100
                val_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_samples * 100
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} ({train_acc:.2f}%) | Val Loss: {avg_val_loss:.4f} ({val_acc:.2f}%)")
        
        # Save checkpoint every 5 epochs
        #if (epoch + 1) % 5 == 0:
        if (epoch + 1) % 1 == 0:
            save_path = f"processed/model_epoch_{epoch+1}_{avg_val_loss:.2f}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}\n")
            

if __name__ == "__main__":
    train_phase1()
