import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_cmapss_data(dataset='FD003', data_dir='data/raw'):
    """
    Load NASA C-MAPSS dataset
    
    Args:
        dataset: Dataset name (FD001, FD002, FD003, or FD004)
        data_dir: Directory containing the raw data files
        
    Returns:
        X_train, y_train, X_test, y_test with CONSISTENT features
    """
    # Column names for C-MAPSS
    index_names = ['unit_id', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # File paths
    train_file = os.path.join(data_dir, f'train_{dataset}.txt')
    test_file = os.path.join(data_dir, f'test_{dataset}.txt')
    rul_file = os.path.join(data_dir, f'RUL_{dataset}.txt')
    
    # Check if files exist
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    
    # Load training data
    train_df = pd.read_csv(
        train_file,
        sep=r'\s+',
        header=None,
        names=col_names,
        index_col=False
    )
    
    # Load test data
    test_df = pd.read_csv(
        test_file,
        sep=r'\s+',
        header=None,
        names=col_names,
        index_col=False
    )
    
    # Load RUL values for test set
    rul_df = pd.read_csv(
        rul_file,
        sep=r'\s+',
        header=None,
        names=['RUL'],
        index_col=False
    )
    
    # Create labels BEFORE feature engineering
    train_df = add_labels(train_df, degradation_threshold=125)
    test_df = add_labels_test(test_df, rul_df, degradation_threshold=125)
    
    # Identify informative sensors from TRAINING data only
    sensors_to_keep = identify_informative_sensors(train_df)
    
    print(f"Selected {len(sensors_to_keep)} sensors: {sensors_to_keep}")
    
    # CRITICAL: Ensure test data has the same sensors
    # Add any missing sensors to test_df with zeros
    for sensor in sensors_to_keep:
        if sensor not in test_df.columns:
            test_df[sensor] = 0.0
    
    # Create windows with SAME features for both train and test
    X_train, y_train = create_windows(
        train_df, 
        sensors_to_keep, 
        window_size=30,
        stride=15
    )
    X_test, y_test = create_windows(
        test_df,
        sensors_to_keep,
        window_size=30,
        stride=15
    )
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # VERIFY they have the same number of features
    assert X_train.shape[2] == X_test.shape[2], \
        f"Feature mismatch! Train: {X_train.shape[2]}, Test: {X_test.shape[2]}"
    
    return X_train, y_train, X_test, y_test


def add_labels(df, degradation_threshold=125):
    """Add RUL and degradation labels to training data"""
    # Calculate RUL for each unit
    df['RUL'] = df.groupby('unit_id')['time_cycles'].transform(
        lambda x: x.max() - x
    )
    # Binary label: 0 = normal, 1 = degrading
    df['label'] = (df['RUL'] <= degradation_threshold).astype(int)
    return df


def add_labels_test(test_df, rul_df, degradation_threshold=125):
    """Add labels to test data using provided RUL values"""
    # Get max cycles for each test unit
    max_cycles = test_df.groupby('unit_id')['time_cycles'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    
    # Add RUL from file
    max_cycles['RUL_end'] = rul_df['RUL'].values
    
    # Merge back to test_df
    test_df = test_df.merge(max_cycles, on='unit_id', how='left')
    
    # Calculate RUL for each timestep
    test_df['RUL'] = test_df['max_cycle'] - test_df['time_cycles'] + test_df['RUL_end']
    
    # Binary label
    test_df['label'] = (test_df['RUL'] <= degradation_threshold).astype(int)
    
    # Clean up
    test_df = test_df.drop(['max_cycle', 'RUL_end'], axis=1)
    
    return test_df


def identify_informative_sensors(df):
    """
    Identify informative sensors from training data
    Returns: List of sensor column names (NO rolling features)
    """
    # Only base sensors, no rolling features
    sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
    
    # Remove constant sensors
    variances = df[sensor_cols].var()
    informative_sensors = variances[variances > 0.01].index.tolist()
    
    # Sort for consistency
    informative_sensors.sort()
    
    return informative_sensors


def create_windows(df, features, window_size=30, stride=15):
    """Create sliding windows from time series data"""
    X, y = [], []
    
    # Verify all features exist in dataframe
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) != len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: Missing features {missing}, using available ones")
        features = available_features
    
    if not features:
        raise ValueError("No features available for windowing!")
    
    for unit_id in df['unit_id'].unique():
        unit_df = df[df['unit_id'] == unit_id].copy()
        
        # Handle missing values
        unit_df[features] = unit_df[features].fillna(0)
        
        values = unit_df[features].values
        labels = unit_df['label'].values
        
        # Create windows
        for i in range(0, len(values) - window_size + 1, stride):
            window = values[i:i+window_size]
            label = labels[i+window_size-1]  # Use last timestep label
            X.append(window)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


# ============================================================================
# SIMPLIFIED PARTITION FUNCTION
# ============================================================================

def partition_data(X, y, num_devices=50, heterogeneity='high'):
    """
    Create balanced partitions - GUARANTEED both classes
    """
    
    # Separate by class
    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]
    
    print(f"\nPartitioning {len(X)} samples:")
    print(f"  Class 0: {len(idx_class_0)}")
    print(f"  Class 1: {len(idx_class_1)}")
    
    np.random.shuffle(idx_class_0)
    np.random.shuffle(idx_class_1)
    
    # Ensure we have enough samples
    min_per_device = 30
    max_devices = min(len(idx_class_0) // min_per_device, 
                      len(idx_class_1) // min_per_device)
    
    if num_devices > max_devices:
        print(f"  Adjusting devices from {num_devices} to {max_devices}")
        num_devices = max(max_devices, 2)
    
    # Split each class equally among devices
    class_0_splits = np.array_split(idx_class_0, num_devices)
    class_1_splits = np.array_split(idx_class_1, num_devices)
    
    device_data = []
    
    for i in range(num_devices):
        # Combine both classes for this device
        device_indices = np.concatenate([class_0_splits[i], class_1_splits[i]])
        np.random.shuffle(device_indices)
        
        device_X = X[device_indices]
        device_y = y[device_indices]
        
        n_class_0 = np.sum(device_y == 0)
        n_class_1 = np.sum(device_y == 1)
        
        print(f"  Device {i}: {len(device_X)} samples (0:{n_class_0}, 1:{n_class_1})")
        
        device_data.append((device_X, device_y))
    
    return device_data