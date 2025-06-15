import numpy as np
import pandas as pd

# Load CSV
df = pd.read_csv("btcusd_1-min_data.csv")

# Drop timestamp if not used
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(np.float32)

# Normalize (optional)
# df = (df - df.mean()) / df.std()

# Input: rows 0..N-2 (to predict next Close)
X = df.iloc[:-1].to_numpy()  # shape (N-1, 5)

# Target: Close price at next minute
Y = df['Close'].iloc[1:].to_numpy()  # shape (N-1,)

# Save as binary
X.tofile("input.bin")
Y.tofile("target.bin")

print(f"Saved input.bin (shape: {X.shape}), target.bin (shape: {Y.shape})")
