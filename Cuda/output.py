import numpy as np
import matplotlib.pyplot as plt
import os

# File paths
pred_file = "output.bin"
true_file = "target.bin"

# Check file existence
if not os.path.exists(pred_file) or not os.path.exists(true_file):
    print("Error: One or both binary files not found.")
    exit(1)

# Load binary data
pred = np.fromfile(pred_file, dtype=np.float32)
true = np.fromfile(true_file, dtype=np.float32)

# Optional: truncate to minimum length if mismatched
N = min(len(pred), len(true))
pred = pred[:N]
true = true[:N]

print(f"Loaded {N} entries")

# Optional: Slice for clarity, e.g., first 10000 points
SAMPLE_COUNT = min(10000, N)
pred_plot = pred[:SAMPLE_COUNT]
true_plot = true[:SAMPLE_COUNT]

# Plot
plt.figure(figsize=(15, 5))
plt.plot(pred_plot, label="Predicted Close", color='orange')
plt.plot(true_plot, label="Actual Close", color='blue', alpha=0.6)
plt.title(f"Predicted vs Actual BTC Close Prices ({SAMPLE_COUNT} samples)")
plt.xlabel("Time (minutes)")
plt.ylabel("BTC Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
