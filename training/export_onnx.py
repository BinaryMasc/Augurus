import torch
import os
from model import FinancialTransformer
from config import MODEL_CONFIG
import shutil
import json
import pickle

def export_to_onnx():
    DEVICE = torch.device("cpu") # Exporting on CPU is standard for portability
    
    # 1. Initialize the model with exact same parameters as training
    model = FinancialTransformer(**MODEL_CONFIG).to(DEVICE)
    
    # 2. Load the trained RL weights
    #checkpoint_path = "processed/model_rl_epoch_2.pt"
    checkpoint_path = "processed/model_epoch_14_6.25.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find {checkpoint_path}")
        return
        
    print(f"Loading weights from {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    except RuntimeError as e:
        print(f"Warning: Checkpoint shape mismatch detected ({e}).")
        print("Model weights could not be loaded due to architecture changes. Exporting raw/random model.")
    model.eval() # MUST set to evaluation mode before export (disables dropout, etc.)
    
    # 3. Create dummy inputs for ONNX tracing
    batch_size = 1
    seq_len = MODEL_CONFIG['seq_len']
    
    dummy_tokens = torch.randint(0, MODEL_CONFIG['vocab_size'], (batch_size, seq_len), dtype=torch.long).to(DEVICE)
    dummy_continuous = torch.randn(batch_size, seq_len, MODEL_CONFIG['num_continuous_features'], dtype=torch.float32).to(DEVICE)
    
    # Define dynamic axes to allow varying batch sizes during inference in C#
    dynamic_axes = {
        'tokens': {0: 'batch_size'},
        'continuous_features': {0: 'batch_size'},
        'next_token_logits': {0: 'batch_size'},
        'action_logits': {0: 'batch_size'},
        'state_values': {0: 'batch_size'}
    }
    
    onnx_path = "../inference/financial_transformer.onnx"
    os.makedirs("../inference", exist_ok=True)
    
    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model, 
        (dummy_tokens, dummy_continuous), 
        onnx_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['tokens', 'continuous_features'],
        output_names=['next_token_logits', 'action_logits', 'state_values'],
        dynamic_axes=dynamic_axes
    )
    
    print(f"Model successfully exported to {onnx_path}")
    
    # 4. Export the Quantizer Bins as JSON for the C# API
    quantizer_pkl_path = "processed/quantizer.pkl"
    if os.path.exists(quantizer_pkl_path):
        print(f"Exporting quantizer bins from {quantizer_pkl_path}...")
        with open(quantizer_pkl_path, 'rb') as f:
            quantizer = pickle.load(f)
        
        # Extract bin edges for the norm_return feature
        edges = quantizer.kbins.bin_edges_[0]
        
        # Replace -inf and inf with large numbers for JSON compatibility
        edges_list = edges.tolist()
        edges_list[0] = -1e9
        edges_list[-1] = 1e9

        json_out_path = "../inference/quantizer_bins.json"
        with open(json_out_path, 'w') as f:
            json.dump({"bin_edges": edges_list}, f)
        
        print(f"Quantizer bins exported to {json_out_path}")
        
        # Also copy the raw pkl for safety
        shutil.copy(quantizer_pkl_path, "../inference/quantizer.pkl")
    else:
        print("Warning: quantizer.pkl not found. Skipping JSON export.")

if __name__ == "__main__":
    export_to_onnx()
