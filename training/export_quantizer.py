import pickle
import json
import os
import sys

# Need to import FinancialQuantizer so pickle can find the class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import FinancialQuantizer

def export_quantizer():
    with open('processed/quantizer.pkl', 'rb') as f:
        quantizer = pickle.load(f)

    # Extract bin edges for feature 0 (norm_return)
    edges = quantizer.kbins.bin_edges_[0]
    
    # Replace -inf and inf with large numbers for JSON compatibility
    edges[0] = -1e9
    edges[-1] = 1e9
    edges_list = edges.tolist()

    os.makedirs('../inference', exist_ok=True)
    with open('../inference/quantizer_bins.json', 'w') as f:
        json.dump({"bin_edges": edges_list}, f)

    print("Exported bin edges to JSON for C# consumption.")

if __name__ == "__main__":
    export_quantizer()
