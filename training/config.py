# Centralized configuration for the FinancialTransformer model
# Change these values to affect all training and export scripts globally (training and testing).

MODEL_CONFIG = {
    "vocab_size": 256,
    "d_model": 64,       # Reduced from 256 to prevent overfitting
    "nhead": 4,           # divide d_model
    "num_layers": 4,      
    "dropout": 0.25,       # Regularization (used during training)
    "num_continuous_features": 7,
    "seq_len": 64
}

# Training-specific regularization parameters
WEIGHT_DECAY = 0.10
GAUSSIAN_NOISE = 0.05

