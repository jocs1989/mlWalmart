import logging

logger = logging.getLogger(__name__)

def feature_engineering(df):
    logger.info("🔧 Starting feature engineering...")

    # Sort by machine and time
    df = df.sort_values(['machineID', 'datetime'])
    logger.info(f"Data sorted. Number of rows: {len(df)}")

    # Create rolling mean and std with a 3-hour window for telemetry variables
    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    for col in telemetry_cols:
        df[f'{col}_rolling_mean_3'] = df.groupby('machineID')[col].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df[f'{col}_rolling_std_3'] = df.groupby('machineID')[col].transform(lambda x: x.rolling(window=3, min_periods=1).std().fillna(0))
    logger.info(f"Rolling features calculated for: {telemetry_cols}")

    # Encode 'model' as numeric category
    df['model'] = df['model'].astype('category').cat.codes
    logger.info("Column 'model' encoded as numeric.")

    # Extract hour of day from datetime
    df['hour'] = df['datetime'].dt.hour
    logger.info("Temporal variable 'hour' created.")

    # Select feature columns (exclude IDs and label)
    exclude_cols = ['machineID', 'datetime', 'failure_in_next_24h']
    feature_cols = [c for c in df.columns if c not in exclude_cols and not c.startswith('error')]  # omitting errors for now
    logger.info(f"Selected columns as features: {feature_cols}")

    # Fill missing values with zero
    X = df[feature_cols].fillna(0)
    y = df['failure_in_next_24h']

    logger.info(f"Features and label separated. Features shape: {X.shape}, Label shape: {y.shape}")
    return X, y
