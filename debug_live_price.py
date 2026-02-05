
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from core.data import loader

print("Testing fetch_live_data for GC=F...")
try:
    data = loader.fetch_live_data('GC=F')
    print(f"Result: {data}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting fetch_market_snapshot...")
try:
    snapshot = loader.fetch_market_snapshot()
    print("Snapshot keys:", snapshot.keys())
    for k, v in snapshot.items():
        print(f"{k}: {v}")
except Exception as e:
    print(f"Snapshot Error: {e}")
