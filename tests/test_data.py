# test_data.py
from data.preprocess import load_cmapss_data, partition_data
import numpy as np

def test_data_pipeline():
    print("Testing data loading...")
    X_train, y_train, X_test, y_test = load_cmapss_data('FD003')
    print(f"✓ Data loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    print("\nTesting data partitioning...")
    device_data = partition_data(X_train, y_train, num_devices=10)
    print(f"✓ Created {len(device_data)} partitions")
    
    # Check class distribution
    for i, (data, labels) in enumerate(device_data[:3]):
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Device {i}: {dict(zip(unique, counts))}")
    
    print("\nData pipeline working correctly!")

if __name__ == "__main__":
    test_data_pipeline()