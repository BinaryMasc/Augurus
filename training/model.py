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

class Time2Vec(nn.Module):
    """
    Time2Vec periodic time representation.
    Maps input features to a vector containing one linear projection and periodic (sine) projections.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))

    def forward(self, x):
        v1 = torch.matmul(x, self.w0) + self.b0
        v2 = torch.sin(torch.matmul(x, self.w) + self.b)
        return torch.cat([v1, v2], dim=-1)

class FinancialTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, nhead=8, num_layers=6, dropout=0.1, num_continuous_features=7, **kwargs):
        super().__init__()
        self.d_model = d_model
        
        # Discrete Token Embedding (0-255 bins)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Feature-wise Continuous Embeddings:
        # Time features (first 4 features: hour_sin, hour_cos, day_sin, day_cos)
        self.time_t2v = Time2Vec(in_features=4, out_features=d_model)
        
        # Other continuous features (norm_return, ema12_dist, ema26_dist)
        self.norm_return_emb = nn.Linear(1, d_model)
        self.ema12_emb = nn.Linear(1, d_model)
        self.ema26_emb = nn.Linear(1, d_model)
        
        # Learned Positional Embedding
        seq_len = kwargs.get('seq_len', 64)
        max_positions = max(seq_len * 2, 2048)
        self.pos_emb = nn.Embedding(max_positions, d_model)
        
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
        self.actor_head = nn.Linear(d_model, 3)
        
        # Critic Head for PPO (Phase 2 RL)
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
        x_tok = self.token_emb(tokens)
        
        # Feature-wise embeddings:
        # Split time features and non-time features
        # continuous_features shape: [batch_size, seq_len, 7]
        # index 0-3: hour_sin, hour_cos, day_sin, day_cos
        # index 4: norm_return
        # index 5: ema12_dist
        # index 6: ema26_dist
        time_feats = continuous_features[:, :, :4]
        norm_ret_feat = continuous_features[:, :, 4:5]
        ema12_feat = continuous_features[:, :, 5:6]
        ema26_feat = continuous_features[:, :, 6:7]
        
        # Project each feature to d_model
        x_time = self.time_t2v(time_feats)
        x_norm_ret = self.norm_return_emb(norm_ret_feat)
        x_ema12 = self.ema12_emb(ema12_feat)
        x_ema26 = self.ema26_emb(ema26_feat)
        
        # Sum continuous feature embeddings
        x_cont = x_time + x_norm_ret + x_ema12 + x_ema26
        
        # Combine embeddings (Additive fusion)
        x = x_tok + x_cont
        
        # Scale by sqrt(d_model)
        x = x * math.sqrt(self.d_model)
        
        # 2. Positional Encoding
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) # [1, seq_len]
        x = x + self.pos_emb(positions)
        
        # 3. Causal Mask to prevent looking into the future
        mask = self.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # 4. Transformer Forward
        output = self.transformer(x, mask=mask, is_causal=True)
        
        # 5. Predictions
        logits = self.fc_out(output)
        action_logits = self.actor_head(output)
        state_values = self.critic_head(output)
        
        return logits, action_logits, state_values

if __name__ == "__main__":
    print("Testing FinancialTransformer architecture...")
    model = FinancialTransformer()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Mock data: Batch of 32 sequences, each 100 candles long
    mock_tokens = torch.randint(0, 256, (32, 100))
    mock_continuous = torch.randn(32, 100, 7)
    
    logits, action_logits, state_values = model(mock_tokens, mock_continuous)
    
    print(f"Tokens Input Shape: {mock_tokens.shape}")
    print(f"Continuous Input Shape: {mock_continuous.shape}")
    print(f"Phase 1 Logits Output Shape: {logits.shape} (batch, seq, vocab)")
    print(f"Phase 2 Action Logits Output Shape: {action_logits.shape} (batch, seq, actions)")
    print(f"Phase 2 State Values Output Shape: {state_values.shape} (batch, seq, 1)")
