# data/preprocess.py

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
        X_train, y_train, X_test, y_test
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
    
    # Feature engineering
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    
    # Create labels (binary: normal vs degrading)
    train_df = add_labels(train_df, degradation_threshold=125)
    test_df = add_labels_test(test_df, rul_df, degradation_threshold=125)
    
    # Remove constant/low-variance sensors
    sensors_to_keep = identify_informative_sensors(train_df)
    
    # Create windows
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
    
    return X_train, y_train, X_test, y_test


def add_features(df):
    """Add rolling statistics and degradation indicators"""
    for sensor in [col for col in df.columns if 'sensor' in col]:
        # Rolling mean and std
        df[f'{sensor}_rolling_mean'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df[f'{sensor}_rolling_std'] = df.groupby('unit_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
    return df


def add_labels(df, degradation_threshold=125):
    """Add RUL and degradation labels to training data"""
    # Calculate RUL for each unit
    df['RUL'] = df.groupby('unit_id')['time_cycles'].transform(
        lambda x: x.max() - x
    )
    # Binary label: 0 = normal, 1 = degrading
    df['label'] = (df['RUL'] <= degradation_threshold).astype(int)
    return df

def identify_informative_sensors(df):
    """Remove sensors with low variance - keep only base sensors for consistency"""
    sensor_cols = [col for col in df.columns if 'sensor' in col and 'rolling' not in col]
    variances = df[sensor_cols].var()
    # Keep sensors with variance > threshold
    informative_sensors = variances[variances > 0.01].index.tolist()
    
    # ONLY use base sensors (no rolling features) to keep consistent dimensions
    return informative_sensors

def create_windows(df, features, window_size=30, stride=15):
    """Create sliding windows from time series data"""
    X, y = [], []
    
    # Ensure features exist in dataframe
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        # Fallback to basic sensor columns
        available_features = [col for col in df.columns if 'sensor' in col][:14]
    
    for unit_id in df['unit_id'].unique():
        unit_df = df[df['unit_id'] == unit_id].copy()
        
        # Handle missing values
        unit_df[available_features] = unit_df[available_features].fillna(0)
        
        values = unit_df[available_features].values
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

def add_labels_test(df, rul_df, degradation_threshold=125):
    """Add RUL and degradation labels to test data using provided RUL values"""
    # Add unit_id index to rul_df
    rul_df = rul_df.reset_index()
    rul_df['unit_id'] = rul_df.index + 1
    
    # # For each unit, calculate RUL for each time cycle
    # def calculate_test_rul(group):
    #     unit_id = group['unit_id'].iloc[0]
    #     max_cycles = group['time_cycles'].max()
    #     final_rul = rul_df[rul_df['unit_id'] == unit_id]['RUL'].iloc[0]
    #     group['RUL'] = final_rul + (max_cycles - group['time_cycles'])
    #     return group
    
    # df = df.groupby('unit_id').apply(calculate_test_rul).reset_index(drop=True)
    
    # Binary label: 0 = normal, 1 = degrading
    df['label'] = (rul_df['RUL'] <= degradation_threshold).astype(int)
    return df

    """Create sliding windows from time series data"""
    X, y = [], []
    
    for unit_id in df['unit_id'].unique():
        unit_df = df[df['unit_id'] == unit_id]
        values = unit_df[features].values
        labels = unit_df['label'].values
        
        for i in range(0, len(values) - window_size + 1, stride):
            window = values[i:i+window_size]
            label = labels[i+window_size-1]  # Use last timestep label
            X.append(window)
            y.append(label)
    
    return np.array(X), np.array(y)