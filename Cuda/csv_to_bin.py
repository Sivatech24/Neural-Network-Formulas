import csv
import struct

# Format: Q = unsigned long long (for timestamp), d = double
fmt = 'Q d d d d d'
record_size = struct.calcsize(fmt)

with open('btcusd_1-min_data.csv', 'r') as infile, open('btcusd_data.bin', 'wb') as outfile:
    reader = csv.DictReader(infile)
    for row in reader:
        try:
            ts = int(float(row['Timestamp']))
            ohlcv = tuple(float(row[col]) for col in ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)'])
            packed = struct.pack(fmt, ts, *ohlcv)
            outfile.write(packed)
        except (KeyError, ValueError):
            continue  # Skip if data is missing or malformed
