# Augurus

Augurus is a financial market prediction system utilizing a Transformer architecture for next-token prediction and RL fine-tuning.

## Project Structure
- **/training**: Python-based training pipeline (PyTorch).
- **/inference**: C#-based API for real-time market predictions (ONNX).


## Augurus Training Pipeline


#### 1. Data Preprocessing
Converts raw Binance OHLCV CSV data into tokenized features and calculates rolling volatility.
```bash
python process_data.py
```
*   **Input**: `data/BTCUSDT_5m_...csv`
*   **Output**: `processed/tokenized_data.csv`, `processed/quantizer.pkl`

### 2. Phase 1: Next-Token Prediction (Pre-training)
Trains the Transformer to predict the next market "token" (price movement bin). This builds the model's understanding of market dynamics.
```bash
python train.py
```
*   **Config**: `SEQ_LEN=64`, `BATCH_SIZE=128`
*   **Output**: `processed/model_epoch_N.pt`

### 3. Phase 2: RL Fine-tuning (Actor-Critic)
Fine-tunes the pre-trained model using Reinforcement Learning to optimize for actual trading PnL (Profit and Loss). (Not necesary for good results)
```bash
python rl_train.py
```
*   **Requires**: A checkpoint from Phase 1 (e.g., `model_epoch_5.pt`).
*   **Output**: `processed/model_rl_epoch_N.pt`

### 4. Export to ONNX
Converts the trained PyTorch model into ONNX format for high-performance inference in the C# `Augurus.Api`.
```bash
python export_onnx.py
```
*   **Output**: `../inference/financial_transformer.onnx`, `../inference/quantizer.pkl`

---

#### Hardware Notes
- **CUDA**: Recommended for `train.py` and `rl_train.py`.
- **Memory**: Reducing `SEQ_LEN` to 64 allows for larger `BATCH_SIZE` (e.g., 128) on consumer GPUs.
