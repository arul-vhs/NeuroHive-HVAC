import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time

# --- Configuration ---
class Config:
    # Data Paths
    OUTPUT_DIR = './output/'
    BUILDING_ID = 7
    METER_TYPE = 1
    TARGET_COLUMN = 'meter_reading' # The column we want to predict

    # Model Hyperparameters
    SEQ_LENGTH = 24       # How many hours of past data to use for prediction
    PREDICTION_HORIZON = 6 # How many hours into the future to predict
    INPUT_FEATURES = -1   # Will be set automatically after loading data
    LSTM_HIDDEN_SIZE = 64
    LSTM_LAYERS = 2
    TCN_CHANNELS = [64, 128] # Number of channels in TCN layers
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.2
    OUTPUT_SIZE = 1       # Predicting only 'meter_reading'

    # Training Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50 # Adjust as needed, start with fewer (e.g., 10) for quick testing
    PATIENCE = 5 # For early stopping
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File Names (Derived from config)
    def __init__(self):
        base_name = f"b{self.BUILDING_ID}_m{self.METER_TYPE}"
        self.train_scaled_file = os.path.join(self.OUTPUT_DIR, f'train_scaled_{base_name}.csv')
        self.val_scaled_file = os.path.join(self.OUTPUT_DIR, f'val_scaled_{base_name}.csv')
        self.test_scaled_file = os.path.join(self.OUTPUT_DIR, f'test_scaled_{base_name}.csv')
        self.feature_scaler_file = os.path.join(self.OUTPUT_DIR, f'feature_scaler_{base_name}.joblib')
        self.target_scaler_file = os.path.join(self.OUTPUT_DIR, f'target_scaler_{base_name}.joblib')
        self.model_save_path = os.path.join(self.OUTPUT_DIR, f'lstm_tcn_forecaster_{base_name}.pth')
        # Adding path for original featured data for analysis
        self.original_featured_file = os.path.join(self.OUTPUT_DIR, f'featured_building_{self.BUILDING_ID}_meter_{self.METER_TYPE}.csv')


cfg = Config()
print(f"Using device: {cfg.DEVICE}")

# --- 1. PyTorch Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, data_df, target_col, seq_length, pred_horizon):
        self.seq_length = seq_length
        self.pred_horizon = pred_horizon
        self.target_col = target_col

        # Make sure target column exists before dropping
        if self.target_col not in data_df.columns:
             raise ValueError(f"Target column '{self.target_col}' not found in DataFrame.")

        self.features = data_df.drop(columns=[target_col]).values
        self.target = data_df[[target_col]].values

        # Update input features count in config ONLY IF not already set positively
        if cfg.INPUT_FEATURES < 0 :
             cfg.INPUT_FEATURES = self.features.shape[1]
             print(f"Dataset created. Input features set to: {cfg.INPUT_FEATURES}")
        # Verify consistency if already set
        elif cfg.INPUT_FEATURES != self.features.shape[1]:
             print(f"Warning: Dataset feature count ({self.features.shape[1]}) differs from config ({cfg.INPUT_FEATURES}). Using dataset count.")
             cfg.INPUT_FEATURES = self.features.shape[1]
        else:
             # Already set and matches, no need to print again.
             pass


    def __len__(self):
        # Total length minus sequence length (for input) minus prediction horizon (for target)
        return len(self.features) - self.seq_length - self.pred_horizon + 1

    def __getitem__(self, idx):
        # Input sequence: features from idx to idx + seq_length
        feature_start = idx
        feature_end = idx + self.seq_length
        input_seq = self.features[feature_start:feature_end]

        # Target sequence: target values from idx + seq_length to idx + seq_length + pred_horizon
        target_start = feature_end
        target_end = target_start + self.pred_horizon
        target_seq = self.target[target_start:target_end]

        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq).squeeze(-1) # Squeeze target last dim

# --- 2. TCN Components ---
# Basic building block for TCN
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# Helper module to remove padding from the end of the sequence
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# The TCN model itself, stacking TemporalBlocks
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i # Exponentially increasing dilation
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- 3. Combined LSTM + TCN Model ---
class LSTM_TCN_Forecaster(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_layers,
                 tcn_channels, tcn_kernel_size, tcn_dropout,
                 output_size, pred_horizon):
        super(LSTM_TCN_Forecaster, self).__init__()
        self.pred_horizon = pred_horizon

        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_layers,
                            batch_first=True, dropout=tcn_dropout if lstm_layers > 1 else 0)

        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)

        combined_features = lstm_hidden_size + tcn_channels[-1]

        self.fc = nn.Linear(combined_features, output_size * pred_horizon)


    def forward(self, x):
        # Input x shape: (batch_size, seq_length, input_size)
        lstm_out, _ = self.lstm(x)
        lstm_last_out = lstm_out[:, -1, :]

        # TCN Path: (batch_size, input_size, seq_length)
        tcn_in = x.permute(0, 2, 1)
        tcn_out = self.tcn(tcn_in)
        # TCN output shape: (batch_size, tcn_channels[-1], seq_length)
        tcn_last_out = tcn_out[:, :, -1]

        combined = torch.cat((lstm_last_out, tcn_last_out), dim=1)
        output = self.fc(combined)

        # Reshape output to (batch_size, pred_horizon, output_size)
        output = output.view(output.size(0), self.pred_horizon, -1) # Assumes output_size=1

        return output.squeeze(-1) # Squeeze last dim if output_size is 1

# --- 4. Training Function ---
def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    print("\nStarting Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(config.EPOCHS):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(config.DEVICE), targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(config.DEVICE), targets.to(config.DEVICE)
                outputs = model(features)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Duration: {epoch_duration:.2f}s - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Validation loss improved. Saved model to {config.model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{config.PATIENCE}")
            if patience_counter >= config.PATIENCE:
                print("Early stopping triggered.")
                break

    print("Training Finished.")
    return history

# --- 5. Evaluation Function ---
# <<< MODIFIED: Added history parameter >>>
def evaluate_model(model, test_loader, criterion, target_scaler, config, history):
    print("\nStarting Evaluation on Test Set...")
    model.load_state_dict(torch.load(config.model_save_path)) # Load best model
    model.eval()

    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss (Scaled): {avg_test_loss:.4f}")

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Inverse transform
    num_samples = all_predictions.shape[0]
    pred_horizon = all_predictions.shape[1]
    predictions_reshaped = all_predictions.reshape(-1, 1)
    targets_reshaped = all_targets.reshape(-1, 1)

    predictions_unscaled = target_scaler.inverse_transform(predictions_reshaped)
    targets_unscaled = target_scaler.inverse_transform(targets_reshaped)

    predictions_unscaled = predictions_unscaled.reshape(num_samples, pred_horizon)
    targets_unscaled = targets_unscaled.reshape(num_samples, pred_horizon)

    # Calculate metrics
    mae_scores = []
    rmse_scores = []
    print("\nMetrics on Unscaled Data (per prediction step):")
    for i in range(pred_horizon):
        mae = mean_absolute_error(targets_unscaled[:, i], predictions_unscaled[:, i])
        rmse = np.sqrt(mean_squared_error(targets_unscaled[:, i], predictions_unscaled[:, i]))
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        print(f"  Horizon Step {i+1}: MAE = {mae:.2f}, RMSE = {rmse:.2f}")

    overall_mae = mean_absolute_error(targets_unscaled.flatten(), predictions_unscaled.flatten())
    overall_rmse = np.sqrt(mean_squared_error(targets_unscaled.flatten(), predictions_unscaled.flatten()))
    print(f"\nOverall Metrics:")
    print(f"  Overall MAE: {overall_mae:.2f}")
    print(f"  Overall RMSE: {overall_rmse:.2f}")

    # Plotting
    # <<< MODIFIED: Change sample_idx here if needed >>>
    sample_idx = 100 # Plot the first sample (can change this to 50, 100 etc.)
    if sample_idx >= len(targets_unscaled):
         print(f"Warning: sample_idx {sample_idx} is out of bounds for plotting ({len(targets_unscaled)} samples). Plotting index 0.")
         sample_idx = 0

    plt.figure(figsize=(12, 6))
    plt.plot(range(pred_horizon), targets_unscaled[sample_idx, :], label='Actual Meter Reading', marker='o')
    plt.plot(range(pred_horizon), predictions_unscaled[sample_idx, :], label='Predicted Meter Reading', marker='x', linestyle='--')
    plt.title(f'Test Set Forecast vs Actuals (Sample {sample_idx}, Horizon={pred_horizon} hrs)')
    plt.xlabel('Hours into Future')
    plt.ylabel('Meter Reading (Unscaled)')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(cfg.OUTPUT_DIR, f'forecast_vs_actual_b{cfg.BUILDING_ID}_m{cfg.METER_TYPE}.png')
    plt.savefig(plot_filename)
    print(f"\nSaved sample prediction plot to: {plot_filename}")
    # plt.show() # Comment out if running in non-interactive environment

    # <<< MODIFIED: Removed history from return statement >>>
    return overall_mae, overall_rmse

# --- 6. Main Execution Block ---
if __name__ == "__main__":
    # Load Data
    try:
        train_df = pd.read_csv(cfg.train_scaled_file, index_col='timestamp', parse_dates=True)
        val_df = pd.read_csv(cfg.val_scaled_file, index_col='timestamp', parse_dates=True)
        test_df = pd.read_csv(cfg.test_scaled_file, index_col='timestamp', parse_dates=True)

        # <<< --- SNIPPET 2 INSERTED HERE --- >>>
        print("\n--- Inspecting Start of Test Set Target ---")
        # Ensure the target column exists before trying to access it
        if cfg.TARGET_COLUMN in test_df.columns:
             print(test_df[[cfg.TARGET_COLUMN]].head(cfg.SEQ_LENGTH + cfg.PREDICTION_HORIZON + 5)) # Show enough data for first sequence + target
        else:
             print(f"Error: Target column '{cfg.TARGET_COLUMN}' not found in test_df.")
        print("--- End Inspect ---")
        # <<< --- END OF SNIPPET 2 --- >>>

        # Load Scalers
        feature_scaler = joblib.load(cfg.feature_scaler_file)
        target_scaler = joblib.load(cfg.target_scaler_file)
        print("Data and scalers loaded successfully.")

    except FileNotFoundError as e:
        print(f"Error loading data/scaler files: {e}")
        print("Please ensure previous steps (loading, feature engineering, scaling) were completed.")
        exit()
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        exit()

    # Create Datasets and DataLoaders
    try:
        train_dataset = TimeSeriesDataset(train_df, cfg.TARGET_COLUMN, cfg.SEQ_LENGTH, cfg.PREDICTION_HORIZON)
        val_dataset = TimeSeriesDataset(val_df, cfg.TARGET_COLUMN, cfg.SEQ_LENGTH, cfg.PREDICTION_HORIZON)
        test_dataset = TimeSeriesDataset(test_df, cfg.TARGET_COLUMN, cfg.SEQ_LENGTH, cfg.PREDICTION_HORIZON)

        if cfg.INPUT_FEATURES == -1:
             print("Error: INPUT_FEATURES was not set by Dataset. Check data loading.")
             exit()
        print(f"Input features detected/verified: {cfg.INPUT_FEATURES}")

        train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True) # Shuffle training data
        val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    except Exception as e:
        print(f"Error creating Datasets/DataLoaders: {e}")
        exit()


    # Initialize Model, Loss, Optimizer
    model = LSTM_TCN_Forecaster(
        input_size=cfg.INPUT_FEATURES,
        lstm_hidden_size=cfg.LSTM_HIDDEN_SIZE,
        lstm_layers=cfg.LSTM_LAYERS,
        tcn_channels=cfg.TCN_CHANNELS,
        tcn_kernel_size=cfg.TCN_KERNEL_SIZE,
        tcn_dropout=cfg.TCN_DROPOUT,
        output_size=cfg.OUTPUT_SIZE,
        pred_horizon=cfg.PREDICTION_HORIZON
    ).to(cfg.DEVICE)

    criterion = nn.MSELoss() # Mean Squared Error is common for regression
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    print("\nModel Architecture:")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params}")

    # Train the Model
    training_history = train_model(model, train_loader, val_loader, criterion, optimizer, cfg)

    # Evaluate the Model
    # <<< MODIFIED: Pass training_history to evaluate_model >>>
    overall_mae, overall_rmse = evaluate_model(model, test_loader, criterion, target_scaler, cfg, training_history)

    # <<< --- SNIPPET 1 INSERTED HERE --- >>>
    try:
        print("\n--- Analyzing Target Variable Scale ---")
        # Load the *original* featured data to see unscaled values
        # Use path from config object
        original_df = pd.read_csv(cfg.original_featured_file)
        if cfg.TARGET_COLUMN in original_df.columns:
             target_values = original_df[cfg.TARGET_COLUMN]
             print(f"Target Variable ('{cfg.TARGET_COLUMN}') Stats (Unscaled):")
             print(f"  Mean:   {target_values.mean():.2f}")
             print(f"  Std Dev:{target_values.std():.2f}")
             print(f"  Min:    {target_values.min():.2f}")
             print(f"  Max:    {target_values.max():.2f}")
             # Calculate MAE as a percentage of the mean (WAPE approx)
             mean_target = target_values.mean()
             if mean_target != 0:
                  wape = (overall_mae / mean_target) * 100
                  print(f"  Overall MAE as % of Mean (WAPE approx): {wape:.2f}%")
             else:
                  print("  Mean target value is zero, cannot calculate WAPE.")
        else:
             print(f"Error: Target column '{cfg.TARGET_COLUMN}' not found in {cfg.original_featured_file}")
        print("--- End Target Analysis ---\n")
        # Optional: free memory if df is large
        # del original_df

    except FileNotFoundError:
         print(f"Error: Could not find original featured data file at {cfg.original_featured_file} for analysis.")
    except Exception as e:
        print(f"Could not perform target analysis: {e}")
    # <<< --- END OF SNIPPET 1 --- >>>


    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(training_history['train_loss'], label='Training Loss')
    plt.plot(training_history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    history_plot_filename = os.path.join(cfg.OUTPUT_DIR, f'training_history_b{cfg.BUILDING_ID}_m{cfg.METER_TYPE}.png')
    plt.savefig(history_plot_filename)
    print(f"Saved training history plot to: {history_plot_filename}")
    # plt.show() # Comment out if running in non-interactive environment

    print("\nScript finished.")