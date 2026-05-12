import pandas as pd
import os
from data_prep import FinancialQuantizer

def process_binance_csv(input_path, output_dir):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Rename Binance columns to standard lowercase
    rename_map = {
        'Open_Time': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns=rename_map)
    
    # Select only needed columns to save memory
    cols_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df = df[cols_to_keep]
    
    print("Initializing FinancialQuantizer...")
    quantizer = FinancialQuantizer(n_bins=256, rolling_vol_window=120)
    
    print("Fitting quantizer and transforming data (this may take a minute)...")
    processed_df = quantizer.fit_transform(df)
    
    os.makedirs(output_dir, exist_ok=True)
    
    quantizer_path = os.path.join(output_dir, 'quantizer.pkl')
    quantizer.save(quantizer_path)
    print(f"Quantizer saved to {quantizer_path}")
    
    output_csv = os.path.join(output_dir, 'tokenized_data.csv')
    processed_df.to_csv(output_csv, index=False)
    print(f"Processed tokenized data saved to {output_csv}")
    
    print("\nSample of processed data:")
    print(processed_df[['timestamp', 'close', 'log_return', 'rolling_vol', 'norm_return', 'token']].head(10))
    print(f"\nTotal rows processed: {len(processed_df)}")

if __name__ == "__main__":
    INPUT_CSV = "data/BTCUSDT_5m_01-01-2022_to_31-12-2025.csv"
    OUTPUT_DIR = "processed"
    
    if os.path.exists(INPUT_CSV):
        process_binance_csv(INPUT_CSV, OUTPUT_DIR)
    else:
        print(f"Error: Could not find {INPUT_CSV}")
