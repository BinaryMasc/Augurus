import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SiGLU(nn.Module):
    """
    Sigmoid-Gated Linear Unit (SwiGLU variant).
    Splits the input in half along the last dimension, applies SiLU to the first half,
    and multiplies it by the second half.
    """
    def forward(self, x):
        gate, val = x.chunk(2, dim=-1)
        return F.silu(gate) * val

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x

class FinancialTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, nhead=8, num_layers=6, dropout=0.1, num_continuous_features=5, **kwargs):
        super().__init__()
        self.d_model = d_model
        
        # Discrete Token Embedding (0-255 bins)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Continuous Feature Embedding (hour_sin, hour_cos, day_sin, day_cos, norm_volume, etc.)
        self.continuous_emb = nn.Linear(num_continuous_features, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Decoder with SiGLU Activation
        # We double dim_feedforward because SiGLU splits the dimension in half
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4 * 2, 
            dropout=dropout,
            batch_first=True,
            activation=SiGLU()
        )
        # Patch linear2 to accept the chunked dimension from SiGLU
        decoder_layer.linear2 = nn.Linear(d_model * 4, d_model)
        
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        
        # Output head for Next-Token Prediction
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Actor Head for PPO (Phase 2 RL)
        # Outputs logits for: Buy, Sell, Hold
        self.actor_head = nn.Linear(d_model, 3)
        
        # Critic Head for PPO (Phase 2 RL)
        # Outputs estimated value of the current state
        self.critic_head = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag. (Causal Mask)"""
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, tokens, continuous_features):
        """
        tokens: [batch_size, seq_len] (Integer tokens 0-255)
        continuous_features: [batch_size, seq_len, num_features] (Floats)
        """
        # 1. Embeddings
        # Convert tokens to dense vectors
        x_tok = self.token_emb(tokens)
        
        # Map continuous features to same dimension
        x_cont = self.continuous_emb(continuous_features)
        
        # Combine embeddings (Additive fusion)
        x = x_tok + x_cont
        
        # Scale by sqrt(d_model)
        x = x * math.sqrt(self.d_model)
        
        # 2. Positional Encoding
        x = x.transpose(0, 1) # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # [batch_size, seq_len, d_model]
        
        # 3. Causal Mask to prevent looking into the future
        seq_len = x.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 4. Transformer Forward
        output = self.transformer(x, mask=mask, is_causal=True)
        
        # 5. Predictions
        # Phase 1 Target: Next-Token logits (predicting the 0-255 token of the next candle)
        logits = self.fc_out(output)
        
        # Phase 2 Target: Trading action probabilities & Value estimation
        action_logits = self.actor_head(output)
        state_values = self.critic_head(output)
        
        return logits, action_logits, state_values

if __name__ == "__main__":
    print("Testing FinancialTransformer architecture...")
    model = FinancialTransformer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Mock data: Batch of 32 sequences, each 100 candles long
    mock_tokens = torch.randint(0, 256, (32, 100))
    mock_continuous = torch.randn(32, 100, 5)
    
    logits, action_logits, state_values = model(mock_tokens, mock_continuous)
    
    print(f"Tokens Input Shape: {mock_tokens.shape}")
    print(f"Continuous Input Shape: {mock_continuous.shape}")
    print(f"Phase 1 Logits Output Shape: {logits.shape} (batch, seq, vocab)")
    print(f"Phase 2 Action Logits Output Shape: {action_logits.shape} (batch, seq, actions)")
    print(f"Phase 2 State Values Output Shape: {state_values.shape} (batch, seq, 1)")
