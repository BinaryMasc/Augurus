import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import FinancialTransformer
from train import FinancialDataset
from config import MODEL_CONFIG
import os
import torch.nn.functional as F

def rl_phase2():
    # 1. Configuration
    SEQ_LEN = MODEL_CONFIG['seq_len']
    BATCH_SIZE = 128 
    EPOCHS = 2
    LR = 1e-5 # Much smaller LR for fine-tuning
    FEE_RATE = 0.0005 # 0.05% exchange fee
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Data
    data_path = "processed/tokenized_data.csv"
    df = pd.read_csv(data_path)
    
    # We use the validation set for RL fine-tuning to avoid overfitting to the pre-training data
    split_idx = int(len(df) * 0.9)
    rl_df = df.iloc[split_idx:].reset_index(drop=True)
    
    dataset = FinancialDataset(rl_df, seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    
    # We also need the raw log_returns to calculate actual PnL reward
    raw_returns = torch.tensor(rl_df['log_return'].values, dtype=torch.float32)
    
    # 3. Initialize Model and load Phase 1 weights
    model = FinancialTransformer(**MODEL_CONFIG).to(DEVICE)
    
    checkpoint_path = "processed/model_epoch_3.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading Phase 1 weights from {checkpoint_path}")
        # strict=False allows us to load the weights even though we just added a new critic_head
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
    else:
        print("Warning: Phase 1 checkpoint not found. Starting from scratch.")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    print(f"\nStarting Phase 2 RL Training (Actor-Critic PPO-style) on device: {DEVICE}")
    for epoch in range(EPOCHS):
        model.train()
        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS} [RL]")
        
        batch_idx = 0
        
        for x_tok, x_cont, _ in progress_bar:
            x_tok, x_cont = x_tok.to(DEVICE), x_cont.to(DEVICE)
            
            # Forward pass
            # We don't use the next-token logits here, only the action probabilities and value estimate
            _, action_logits, state_values = model(x_tok, x_cont)
            
            # Action probabilities (0: Hold, 1: Buy, 2: Sell)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
            
            # Calculate Rewards based on sampled actions
            positions = torch.zeros_like(actions, dtype=torch.float32)
            positions[actions == 1] = 1.0
            positions[actions == 2] = -1.0
            
            # Calculate transition penalties (fees)
            # You pay a fee every time your position changes
            padded_positions = torch.cat([torch.zeros(positions.shape[0], 1).to(DEVICE), positions[:, :-1]], dim=1)
            trades = torch.abs(positions - padded_positions)
            fee_penalties = trades * FEE_RATE
            
            # Extract actual market returns for this batch
            batch_start = batch_idx * BATCH_SIZE
            returns_batch = []
            for i in range(BATCH_SIZE):
                start = batch_start + i + 1
                end = start + SEQ_LEN
                returns_batch.append(raw_returns[start:end])
                
            actual_returns = torch.stack(returns_batch).to(DEVICE)
            
            # Reward calculation: PnL - Fees
            step_rewards = (positions * actual_returns) - fee_penalties
            
            # Advantage for Actor (Reward - Critic Baseline)
            advantages = step_rewards - state_values.squeeze(-1).detach()
            
            # Loss calculations
            critic_loss = F.mse_loss(state_values.squeeze(-1), step_rewards)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs * advantages).mean()
            entropy = dist.entropy().mean()
            
            # Combined Loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_reward += step_rewards.sum().item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            progress_bar.set_postfix(reward=f"{step_rewards.sum().item():.4f}", loss=f"{loss.item():.4f}")
            
            batch_idx += 1
            
        print(f"Epoch {epoch+1} Summary | Total PnL (Log-Returns): {total_reward:.4f} | Actor Loss: {total_actor_loss/len(loader):.4f}")
        
        torch.save(model.state_dict(), f"processed/model_rl_epoch_{epoch+1}.pt")
        print(f"RL Checkpoint saved to processed/model_rl_epoch_{epoch+1}.pt\n")

if __name__ == "__main__":
    rl_phase2()
