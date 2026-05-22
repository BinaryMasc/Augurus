import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os
from model import FinancialTransformer
from config import MODEL_CONFIG

def evaluate_checkpoint(checkpoint_path):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 1. Load Data
    data_path = 'processed/tokenized_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    split_idx = int(len(df) * 0.9)
    
    # Store columns as numpy arrays
    cont_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'norm_return', 'ema12_dist', 'ema26_dist']
    tokens_all = df['token'].values
    continuous_all = df[cont_cols].values.astype(np.float32)

    # 2. Load Quantizer
    quantizer_path = 'processed/quantizer.pkl'
    if os.path.exists(quantizer_path):
        with open(quantizer_path, 'rb') as f:
            quantizer = pickle.load(f)
        bin_edges = quantizer.kbins.bin_edges_[0]
    else:
        print("Warning: quantizer.pkl not found. Using uniform bin edges.")
        bin_edges = np.linspace(-3.0, 3.0, 257)

    # Get bin centers/values to convert tokens back to returns
    bin_edges_expanded = np.copy(bin_edges)
    bin_edges_expanded[0] = -1e9
    bin_edges_expanded[-1] = 1e9
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_values_tensor = torch.tensor(bin_values, dtype=torch.float32, device=DEVICE)

    # 3. Load Model
    model = FinancialTransformer(**MODEL_CONFIG).to(DEVICE)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    def run_eval(start_idx, end_idx, name, max_batches=100, batch_size=256):
        total_loss = 0.0
        total_samples = 0
        
        step_correct = np.zeros(5)
        step_samples = np.zeros(5)
        
        true_up_count = 0
        pred_up_count = 0
        
        seq_len = MODEL_CONFIG['seq_len']
        
        # Determine number of possible starting indices in this split
        # We need seq_len elements, plus up to 5 future tokens
        valid_start_indices = np.arange(start_idx, end_idx - seq_len - 5)
        
        # Limit to max_batches
        num_samples = len(valid_start_indices)
        num_batches = min(max_batches, int(np.ceil(num_samples / batch_size)))
        
        print(f"Evaluating {name} set ({num_batches} batches of size {batch_size})...")
        
        with torch.no_grad():
            for b in range(num_batches):
                b_start = b * batch_size
                b_end = min(b_start + batch_size, num_samples)
                curr_batch_size = b_end - b_start
                if curr_batch_size <= 0:
                    break
                
                batch_indices = valid_start_indices[b_start:b_end]
                
                # Slicing the whole batch as NumPy matrices
                x_tok_np = np.zeros((curr_batch_size, seq_len), dtype=np.int64)
                x_cont_np = np.zeros((curr_batch_size, seq_len, len(cont_cols)), dtype=np.float32)
                y_future_np = np.zeros((curr_batch_size, 5), dtype=np.int64)
                
                for idx_in_batch, start_idx_val in enumerate(batch_indices):
                    x_tok_np[idx_in_batch] = tokens_all[start_idx_val : start_idx_val + seq_len]
                    x_cont_np[idx_in_batch] = continuous_all[start_idx_val : start_idx_val + seq_len]
                    y_future_np[idx_in_batch] = tokens_all[start_idx_val + seq_len : start_idx_val + seq_len + 5]
                
                x_tok = torch.tensor(x_tok_np, dtype=torch.long, device=DEVICE)
                x_cont = torch.tensor(x_cont_np, dtype=torch.float32, device=DEVICE)
                y_future = torch.tensor(y_future_np, dtype=torch.long, device=DEVICE)
                
                curr_tokens = x_tok.clone()
                curr_continuous = x_cont.clone()
                
                for step in range(5):
                    logits, _, _ = model(curr_tokens, curr_continuous)
                    target_tokens = y_future[:, step]
                    
                    if step == 0:
                        loss = criterion(logits[:, -1, :], target_tokens)
                        total_loss += loss.item() * curr_batch_size
                        total_samples += curr_batch_size
                    
                    probs = torch.softmax(logits[:, -1, :], dim=-1)
                    expected_norm_return = (probs * bin_values_tensor).sum(dim=-1)
                    
                    pred_up = expected_norm_return > 0
                    true_up = target_tokens > 127
                    
                    if step == 0:
                        true_up_count += true_up.sum().item()
                        pred_up_count += pred_up.sum().item()

                    step_correct[step] += (pred_up == true_up).sum().item()
                    step_samples[step] += curr_batch_size
                    
                    # Quantize expected return to token for next step
                    next_tokens = np.digitize(expected_norm_return.cpu().numpy(), bin_edges_expanded) - 1
                    next_tokens = np.clip(next_tokens, 0, 255)
                    next_tokens_tensor = torch.tensor(next_tokens, dtype=torch.long, device=DEVICE).unsqueeze(1)
                    
                    # Roll features left
                    curr_tokens = torch.cat([curr_tokens[:, 1:], next_tokens_tensor], dim=1)
                    
                    # Roll continuous features left
                    next_cont_features = torch.zeros(curr_batch_size, 1, MODEL_CONFIG['num_continuous_features'], device=DEVICE)
                    next_cont_features[:, 0, 4] = expected_norm_return
                    curr_continuous = torch.cat([curr_continuous[:, 1:], next_cont_features], dim=1)
            
        avg_loss = total_loss / total_samples
        step_accs = (step_correct / step_samples) * 100
        
        print(f"\n=== {name} Dataset Evaluation ===")
        print(f"Next-Token Cross-Entropy Loss: {avg_loss:.4f}")
        print(f"Step 1 (Next Candle) Directional Accuracy: {step_accs[0]:.2f}%")
        print(f"Step 2 Directional Accuracy: {step_accs[1]:.2f}%")
        print(f"Step 3 Directional Accuracy: {step_accs[2]:.2f}%")
        print(f"Step 4 Directional Accuracy: {step_accs[3]:.2f}%")
        print(f"Step 5 Directional Accuracy: {step_accs[4]:.2f}%")
        
        pred_up_pct = (pred_up_count / total_samples) * 100
        true_up_pct = (true_up_count / total_samples) * 100
        print(f"Directional Bias (Predicted Up %): {pred_up_pct:.2f}% (Actual Up %: {true_up_pct:.2f}%)")

    # Evaluate Train
    run_eval(0, split_idx, "TRAIN", max_batches=100)
    
    # Evaluate Val
    run_eval(split_idx, len(df), "VALIDATION", max_batches=100)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="processed/model_epoch_14_6.25.pt", help="Path to checkpoint model")
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint)
