import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split # Not for time series split directly
from sklearn.preprocessing import MinMaxScaler
import joblib # To save the scaler objects

# --- Configuration ---
OUTPUT_DIR = './output/' # Directory containing featured data and where outputs will be saved
BUILDING_ID_TO_PROCESS = 7 # Use the SAME building ID
METER_TYPE_TO_PROCESS = 1 # Use the SAME meter type
INPUT_FILENAME = os.path.join(OUTPUT_DIR, f'featured_building_{BUILDING_ID_TO_PROCESS}_meter_{METER_TYPE_TO_PROCESS}.csv')

# Split ratios for Train, Validation, Test sets
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
# TEST_RATIO is implied (1 - TRAIN_RATIO - VALIDATION_RATIO)

TARGET_COLUMN = 'meter_reading' # The column we want to predict

# --- Load Feature-Engineered Data ---
print(f"Loading feature-engineered data from: {INPUT_FILENAME}")
try:
    df = pd.read_csv(INPUT_FILENAME, index_col='timestamp', parse_dates=True)
    print(f"Loaded data shape: {df.shape}")
    # Ensure frequency is set if possible (should be 'H')
    if df.index.freq is None:
       try:
           df = df.asfreq('h') # Use 'h' for hourly
           print("Set DatetimeIndex frequency to 'h'.")
       except ValueError:
           print("Warning: Could not set frequency automatically. Proceeding without.")
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILENAME}")
    print("Please ensure you ran the feature engineering script successfully.")
    exit()
except Exception as e:
    print(f"An error occurred loading the data: {e}")
    exit()

if df.empty:
    print("Error: Loaded dataframe is empty.")
    exit()

# --- Data Splitting (Chronological) ---
print("Splitting data into Train, Validation, and Test sets chronologically...")
n_total = len(df)
n_train = int(n_total * TRAIN_RATIO)
n_validation = int(n_total * VALIDATION_RATIO)
n_test = n_total - n_train - n_validation # Remaining data for testing

print(f"Total samples: {n_total}")
print(f"Training samples: {n_train} (until index {n_train-1})")
print(f"Validation samples: {n_validation} (from index {n_train} to {n_train + n_validation - 1})")
print(f"Test samples: {n_test} (from index {n_train + n_validation} to end)")

# Perform the split
train_df = df.iloc[:n_train]
val_df = df.iloc[n_train : n_train + n_validation]
test_df = df.iloc[n_train + n_validation :]

# Verify splits
print(f"Train set shape: {train_df.shape}")
print(f"Validation set shape: {val_df.shape}")
print(f"Test set shape: {test_df.shape}")
assert len(train_df) + len(val_df) + len(test_df) == n_total, "Split lengths don't match total length!"

# --- Data Scaling ---
print("Scaling data using MinMaxScaler...")

# Identify feature columns (all columns except the target)
feature_columns = [col for col in df.columns if col != TARGET_COLUMN]

# Initialize Scalers
# One scaler for features, one for the target (important for inverse transform later)
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit Scalers ONLY on Training Data
print("Fitting scalers on training data...")
# Fit feature scaler on training features
feature_scaler.fit(train_df[feature_columns])
# Fit target scaler on training target (needs reshaping for single feature)
target_scaler.fit(train_df[[TARGET_COLUMN]]) # Use double brackets to keep it as a DataFrame/2D array

# Apply Scalers to all sets (Train, Validation, Test)
print("Transforming Train, Validation, and Test sets...")
# Transform features
train_features_scaled = feature_scaler.transform(train_df[feature_columns])
val_features_scaled = feature_scaler.transform(val_df[feature_columns])
test_features_scaled = feature_scaler.transform(test_df[feature_columns])

# Transform target
train_target_scaled = target_scaler.transform(train_df[[TARGET_COLUMN]])
val_target_scaled = target_scaler.transform(val_df[[TARGET_COLUMN]])
test_target_scaled = target_scaler.transform(test_df[[TARGET_COLUMN]])

# --- Combine Scaled Data (Optional, but convenient for saving/loading) ---
# Create new DataFrames with scaled data and original index
train_scaled_df = pd.DataFrame(train_features_scaled, columns=feature_columns, index=train_df.index)
train_scaled_df[TARGET_COLUMN] = train_target_scaled

val_scaled_df = pd.DataFrame(val_features_scaled, columns=feature_columns, index=val_df.index)
val_scaled_df[TARGET_COLUMN] = val_target_scaled

test_scaled_df = pd.DataFrame(test_features_scaled, columns=feature_columns, index=test_df.index)
test_scaled_df[TARGET_COLUMN] = test_target_scaled

# --- Save Scaled Data and Scalers ---
print("Saving scaled data and scalers...")

# Save scaled dataframes
train_scaled_filename = os.path.join(OUTPUT_DIR, f'train_scaled_b{BUILDING_ID_TO_PROCESS}_m{METER_TYPE_TO_PROCESS}.csv')
val_scaled_filename = os.path.join(OUTPUT_DIR, f'val_scaled_b{BUILDING_ID_TO_PROCESS}_m{METER_TYPE_TO_PROCESS}.csv')
test_scaled_filename = os.path.join(OUTPUT_DIR, f'test_scaled_b{BUILDING_ID_TO_PROCESS}_m{METER_TYPE_TO_PROCESS}.csv')

train_scaled_df.to_csv(train_scaled_filename)
val_scaled_df.to_csv(val_scaled_filename)
test_scaled_df.to_csv(test_scaled_filename)
print(f"Saved scaled training data to: {train_scaled_filename}")
print(f"Saved scaled validation data to: {val_scaled_filename}")
print(f"Saved scaled test data to: {test_scaled_filename}")

# Save the scalers
feature_scaler_filename = os.path.join(OUTPUT_DIR, f'feature_scaler_b{BUILDING_ID_TO_PROCESS}_m{METER_TYPE_TO_PROCESS}.joblib')
target_scaler_filename = os.path.join(OUTPUT_DIR, f'target_scaler_b{BUILDING_ID_TO_PROCESS}_m{METER_TYPE_TO_PROCESS}.joblib')

joblib.dump(feature_scaler, feature_scaler_filename)
joblib.dump(target_scaler, target_scaler_filename)
print(f"Saved feature scaler to: {feature_scaler_filename}")
print(f"Saved target scaler to: {target_scaler_filename}")

# Display sample of scaled training data
print("\nSample of scaled training data:")
print(train_scaled_df.head())
print(f"\nMin value in scaled training data (should be close to 0): {train_scaled_df.values.min()}")
print(f"Max value in scaled training data (should be close to 1): {train_scaled_df.values.max()}")