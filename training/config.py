# Centralized configuration for the FinancialTransformer model
# Change these values to affect all training and export scripts globally (training and testing).

MODEL_CONFIG = {
    "vocab_size": 256,
    "d_model": 64,       # Reduced from 256 to prevent overfitting
    "nhead": 8,           # divide d_model
    "num_layers": 4,      
    "dropout": 0.1,       # Regularization (used during training)
    "num_continuous_features": 5,
    "seq_len": 64
}
