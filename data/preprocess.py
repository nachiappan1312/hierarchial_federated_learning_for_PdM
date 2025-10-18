import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
DATASET_FOLDER = 'data/CMAPSSData'

def load_cmapss_data(dataset='FD001'):
    """
    Load NASA C-MAPSS dataset
    Download from: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
    dataset -  https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
    """
    # Column names for C-MAPSS
    index_names = ['unit_id', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    # Load training data
    train_df = pd.read_csv(
        f'{DATASET_FOLDER}/train_{dataset}.txt',
        sep=' ',
        header=None,
        names=col_names
    )
    
    # Load test data
    test_df = pd.read_csv(
        f'{DATASET_FOLDER}/test_{dataset}.txt',
        sep=' ',
        header=None,
        names=col_names
    )
    
    # Load RUL values for test set
    rul_df = pd.read_csv(
        f'{DATASET_FOLDER}/RUL_{dataset}.txt',
        sep=' ',
        header=None,
        names=['RUL']
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

def identify_informative_sensors(df):
    """Remove sensors with low variance"""
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    variances = df[sensor_cols].var()
    # Keep sensors with variance > threshold
    informative_sensors = variances[variances > 0.01].index.tolist()
    return informative_sensors

def create_windows(df, features, window_size=30, stride=15):
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