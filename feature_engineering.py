import pandas as pd
import numpy as np
import os

# --- Configuration ---
OUTPUT_DIR = './output/' # Directory containing processed data and where feature-engineered data will be saved
BUILDING_ID_TO_PROCESS = 7 # Use the SAME building ID as in the previous step
METER_TYPE_TO_PROCESS = 1 # Use the SAME meter type as in the previous step
INPUT_FILENAME = os.path.join(OUTPUT_DIR, f'processed_building_{BUILDING_ID_TO_PROCESS}_meter_{METER_TYPE_TO_PROCESS}.csv')
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, f'featured_building_{BUILDING_ID_TO_PROCESS}_meter_{METER_TYPE_TO_PROCESS}.csv')

# --- Load Processed Data ---
print(f"Loading processed data from: {INPUT_FILENAME}")
try:
    df = pd.read_csv(INPUT_FILENAME, index_col='timestamp', parse_dates=True)
    print(f"Loaded data shape: {df.shape}")
    # Ensure index is a DatetimeIndex with frequency if possible (should be 'H' from last step)
    if df.index.freq is None:
        df = df.asfreq('H') # Try to set hourly frequency if missing
        print("Set DatetimeIndex frequency to 'H'.")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILENAME}")
    print("Please ensure you ran the previous data loading/processing script successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

# Check for large gaps after loading (optional but good practice)
if df.isnull().values.any():
    print("Warning: Found NaN values after loading. Interpolating again.")
    df = df.interpolate(method='time') # Time-based interpolation for gaps
    df = df.fillna(method='ffill').fillna(method='bfill') # Fill remaining at edges


# --- Feature Engineering ---
print("Starting feature engineering...")

# 1. Time-Based Features
print("Creating time-based features...")
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek # Monday=0, Sunday=6
df['dayofyear'] = df.index.dayofyear
df['month'] = df.index.month
df['weekofyear'] = df.index.isocalendar().week.astype(int) # Use isocalendar for week
df['year'] = df.index.year
df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0) # Saturday or Sunday

# 2. Lag Features (Past Values)
# Create lags for the target variable ('meter_reading') and key weather features
print("Creating lag features...")
target_col = 'meter_reading'
weather_cols_for_lags = ['air_temperature', 'dew_temperature', 'wind_speed']
lag_periods = [1, 2, 3, 6, 12, 24] # Lag hours (e.g., 1hr ago, 2hrs ago, ..., 24hrs ago)

for lag in lag_periods:
    # Lag for target
    df[f'{target_col}_lag{lag}'] = df[target_col].shift(lag)
    # Lags for weather features
    for col in weather_cols_for_lags:
        if col in df.columns: # Check if column exists
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 3. Rolling Window Features (Optional, but often helpful)
# Calculate rolling mean and std dev for target and air temp
print("Creating rolling window features...")
rolling_windows = [3, 6, 12, 24] # Window sizes in hours

for window in rolling_windows:
    # Rolling features for target
    df[f'{target_col}_roll_mean{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
    df[f'{target_col}_roll_std{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
    # Rolling features for air temperature
    if 'air_temperature' in df.columns:
        df[f'air_temp_roll_mean{window}'] = df['air_temperature'].rolling(window=window, min_periods=1).mean()
        df[f'air_temp_roll_std{window}'] = df['air_temperature'].rolling(window=window, min_periods=1).std()

# 4. Cyclical Feature Encoding (Sine/Cosine Transformation)
# This helps models understand the cyclical nature of time features (e.g., hour 23 is close to hour 0)
print("Encoding cyclical features...")

# Hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# Day of Week
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)

# Month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

# Day of Year
df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 366.0) # Use 366 for leap years safety
df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 366.0)

# Drop original cyclical features after encoding
# df = df.drop(columns=['hour', 'dayofweek', 'month', 'dayofyear']) # Keep originals for now, can drop later if needed

# --- Clean Up After Feature Creation ---
# Lag and rolling features introduce NaNs at the beginning of the series
initial_nan_count = df.isnull().sum().sum()
print(f"Total NaN values before final cleanup: {initial_nan_count}")
if initial_nan_count > 0:
    # Drop rows with NaN values created by shifts/rolls (simplest approach)
    # These rows lack the historical context needed for the features
    df = df.dropna()
    print(f"Dropped rows with NaN values introduced by lags/rolling features. New shape: {df.shape}")
else:
    print("No NaN values introduced, or they were already handled.")

if df.empty:
   print("Error: DataFrame is empty after feature engineering and NaN cleanup.")
   print("This might happen if the initial dataset was too short for the specified lags/rolling windows.")
   exit()

# --- Save Feature-Engineered Data ---
df.to_csv(OUTPUT_FILENAME)
print(f"Feature-engineered data saved to: {OUTPUT_FILENAME}")
print(f"Final data shape: {df.shape}")
print("\nSample of feature-engineered data (first 5 rows):")
print(df.head())
print("\nColumns in the final dataframe:")
print(df.columns.tolist())