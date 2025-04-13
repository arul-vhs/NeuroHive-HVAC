import pandas as pd
import numpy as np
import os

# --- Configuration ---
DATA_DIR = './data/' # Make sure this points to where you saved the CSVs
OUTPUT_DIR = './output/' # Directory to save processed data
BUILDING_ID_TO_PROCESS = 7 # Let's start by focusing on one building first
METER_TYPE_TO_PROCESS = 1 # 0=electricity, 1=chilledwater, 2=steam, 3=hotwater. Chilled water (1) is often HVAC related.

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
print("Loading data...")
try:
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), parse_dates=['timestamp'])
    weather_train_df = pd.read_csv(os.path.join(DATA_DIR, 'weather_train.csv'), parse_dates=['timestamp'])
    building_meta_df = pd.read_csv(os.path.join(DATA_DIR, 'building_metadata.csv'))
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print(f"Please ensure train.csv, weather_train.csv, and building_metadata.csv are in the '{DATA_DIR}' directory.")
    exit() # Exit if files aren't found

# --- Filter for a Specific Building and Meter ---
# --- Filter for a Specific Building and Meter ---
print(f"Filtering for Building ID: {BUILDING_ID_TO_PROCESS} and Meter Type: {METER_TYPE_TO_PROCESS}")
train_df = train_df[(train_df['building_id'] == BUILDING_ID_TO_PROCESS) & (train_df['meter'] == METER_TYPE_TO_PROCESS)].copy() # Add .copy() to avoid SettingWithCopyWarning

if train_df.empty:
    print(f"No data found for Building ID {BUILDING_ID_TO_PROCESS} and Meter {METER_TYPE_TO_PROCESS}. Try a different building or meter type.")
    exit()

# --- Merge Building Metadata to get site_id for the filtered train_df ---
print("Merging building metadata to get site_id...")
# Select only necessary columns from building_meta_df
building_info = building_meta_df.loc[building_meta_df['building_id'] == BUILDING_ID_TO_PROCESS, ['building_id', 'site_id']].copy()

# Ensure site_id is present in building_info
if 'site_id' not in building_info.columns or building_info.empty:
     print(f"Error: Could not find site_id for building {BUILDING_ID_TO_PROCESS} in building_metadata.csv")
     exit()
     
site_id = building_info['site_id'].iloc[0] # Get the site_id for filtering weather data
print(f"Building {BUILDING_ID_TO_PROCESS} belongs to Site ID: {site_id}")

# Merge site_id onto the training data
train_df = pd.merge(train_df, building_info[['building_id', 'site_id']], on='building_id', how='left')

# Check if merge was successful and site_id is present
if 'site_id' not in train_df.columns:
    print("Error: Failed to merge site_id into train_df.")
    exit()
if train_df['site_id'].isnull().any():
    print("Error: site_id column contains null values after merge.")
    exit()

# --- Merge Weather Data ---
print(f"Filtering weather data for Site ID: {site_id}")
weather_train_df = weather_train_df[weather_train_df['site_id'] == site_id].copy() # Add .copy()

if weather_train_df.empty:
    print(f"No weather data found for Site ID: {site_id}. Cannot proceed with merge.")
    exit()
    
# Align timestamps before merging - crucial for time series!
# Both dataframes must be sorted by timestamp.
print("Sorting dataframes by timestamp...")
train_df = train_df.sort_values('timestamp')
weather_train_df = weather_train_df.sort_values('timestamp')

# Merge weather data onto the training data using merge_asof
print("Performing merge_asof with weather data...")
merged_df = pd.merge_asof(
    train_df, # Now contains 'timestamp', 'meter_reading', 'building_id', 'meter', 'site_id'
    weather_train_df, # Contains 'timestamp', 'site_id', and weather columns
    on='timestamp',
    by='site_id', # Match rows with the same site_id
    direction='nearest', # Find the closest timestamp
    tolerance=pd.Timedelta('1 hour') # Allow up to 1hr difference
)

print(f"Initial merged data shape: {merged_df.shape}")

# --- Basic Data Cleaning & Feature Selection ---
# (The rest of the script from here onwards should be okay)
print("Cleaning data...")
if train_df.empty:
    print(f"No data found for Building ID {BUILDING_ID_TO_PROCESS} and Meter {METER_TYPE_TO_PROCESS}. Try a different building or meter type.")
    # You might want to explore which buildings HAVE meter type 1 data first.
    # Example: print(train_df[train_df['meter']==1]['building_id'].unique())
    exit()

# --- Merge Building Metadata ---
# Get the site_id for the selected building
site_id = building_meta_df.loc[building_meta_df['building_id'] == BUILDING_ID_TO_PROCESS, 'site_id'].iloc[0]
building_info = building_meta_df[building_meta_df['building_id'] == BUILDING_ID_TO_PROCESS]

# Merge training data with building metadata (optional but can be useful)
# train_df = pd.merge(train_df, building_info, on='building_id', how='left') # Can add later if needed

# --- Merge Weather Data ---
# Filter weather data for the correct site and time range
weather_train_df = weather_train_df[weather_train_df['site_id'] == site_id]

# Align timestamps before merging - crucial for time series!
# We'll merge based on nearest timestamp within an hour tolerance.
# Pandas merge_asof is great for this. Both dataframes must be sorted by timestamp.
train_df = train_df.sort_values('timestamp')
weather_train_df = weather_train_df.sort_values('timestamp')

# Merge weather data onto the training data
# Use merge_asof to find the closest weather reading for each meter reading timestamp
merged_df = pd.merge_asof(
    train_df,
    weather_train_df,
    on='timestamp',
    by='site_id', # Make sure site_id matches
    direction='nearest', # Find the closest timestamp
    tolerance=pd.Timedelta('1 hour') # Allow up to 1hr difference
)

print(f"Initial merged data shape: {merged_df.shape}")

# --- Basic Data Cleaning & Feature Selection ---
print("Cleaning data...")

# Drop redundant columns or those not needed immediately
columns_to_drop = ['site_id', 'meter'] # We already filtered by these
merged_df = merged_df.drop(columns=columns_to_drop, errors='ignore')

# Handle potential missing values after merge (e.g., weather data gaps)
# Let's check missing percentages first
missing_percentage = merged_df.isnull().sum() * 100 / len(merged_df)
print("Missing Value Percentage per Column:\n", missing_percentage[missing_percentage > 0])

# Simple forward fill for weather features - common for time series
weather_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
                'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
for col in weather_cols:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(method='ffill').fillna(method='bfill') # Forward fill, then back fill

# Check missing meter readings (our target) - we might drop these rows if few
if merged_df['meter_reading'].isnull().any():
    print(f"Warning: Missing {merged_df['meter_reading'].isnull().sum()} meter readings.")
    # For now, let's keep them, but we might need to drop or impute later if forecasting this column
    # merged_df = merged_df.dropna(subset=['meter_reading']) # Option to drop rows with missing target

# Set timestamp as index - standard practice for time series analysis
merged_df = merged_df.set_index('timestamp')

# Select final columns for initial forecasting task
# Target: 'meter_reading' (chilled water usage - proxy for cooling load)
# Features: Weather data, maybe time features (we'll add these next)
final_columns = ['meter_reading', 'air_temperature', 'dew_temperature', 'sea_level_pressure', 'wind_speed', 'cloud_coverage']
# Filter only columns that actually exist in the dataframe
final_columns = [col for col in final_columns if col in merged_df.columns]
processed_df = merged_df[final_columns].copy()

# Optional: Resample data to a consistent frequency (e.g., hourly)
# The data is mostly hourly, but resampling ensures consistency
processed_df = processed_df.resample('H').mean() # Use mean aggregation for hourly resampling
print(f"Resampled to hourly frequency. Shape: {processed_df.shape}")

# Handle NaNs introduced by resampling (if gaps existed)
processed_df = processed_df.interpolate(method='time') # Time-based interpolation for gaps
processed_df = processed_df.fillna(method='bfill') # Backfill any remaining NaNs at the start

# --- Save Processed Data ---
output_filename = os.path.join(OUTPUT_DIR, f'processed_building_{BUILDING_ID_TO_PROCESS}_meter_{METER_TYPE_TO_PROCESS}.csv')
processed_df.to_csv(output_filename)
print(f"Processed data saved to: {output_filename}")

# Display first few rows of the result
print("\nSample of processed data:")
print(processed_df.head())