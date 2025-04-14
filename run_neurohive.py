# run_neurohive.py (Enhanced Rule-Based Controller + Enhanced PSO)

import numpy as np
import pandas as pd
import torch
import joblib
import os
import gymnasium as gym
# from stable_baselines3 import SAC # No RL agent
import warnings

# --- Import Custom Modules ---
try:
    from hvac_env import HVACEnv, ForecastConfig, COMFORT_LOW, COMFORT_HIGH, check_basic_rules, COOLING_POWER_LOW, COOLING_POWER_HIGH # Import power constants
except ImportError: print("ERROR: Failed to import HVACEnv..."); exit()
try:
    from pso_controller import Swarm as PSOSwarm, zone_params, num_zones
    from multi_zone_sim import simulate_multi_zone_step
except ImportError: print("ERROR: Failed to import from pso_controller/multi_zone_sim."); exit()
try:
    from train_forecaster import LSTM_TCN_Forecaster
except ImportError: print("ERROR: Failed to import LSTM_TCN_Forecaster."); exit()

# --- Configuration ---
SIMULATION_STEPS = 24 * 7
START_OFFSET_STEPS = 2000

# Paths (Keep automatic/fallback logic as before)
try: cfg_paths = ForecastConfig(); DATA_CSV=cfg_paths.original_featured_file; FORECAST_MODEL_PATH=cfg_paths.model_save_path; FEATURE_SCALER_PATH=cfg_paths.feature_scaler_file; TARGET_SCALER_PATH=cfg_paths.target_scaler_file; BUILDING_ID=cfg_paths.BUILDING_ID; METER_TYPE=cfg_paths.METER_TYPE
except Exception: print("Warning: Using explicit paths."); output_dir='./output/'; base_name="b7_m1"; BUILDING_ID=7; METER_TYPE=1; DATA_CSV=os.path.join(output_dir,f'featured_building_{base_name}.csv'); FORECAST_MODEL=os.path.join(output_dir,f'lstm_tcn_forecaster_{base_name}.pth'); FEATURE_SCALER=os.path.join(output_dir,f'feature_scaler_{base_name}.joblib'); TARGET_SCALER=os.path.join(output_dir,f'target_scaler_{base_name}.joblib')

# PSO Parameters
PSO_PARTICLES = 30
PSO_MAX_ITER = 30

# Output Log File
LOG_FILE = f"./neurohive_simulation_log_ENHANCED_RULE_PSO_b{BUILDING_ID}m{METER_TYPE}.csv" # Updated log name

# Rule-Based Controller Parameters (More nuanced)
RULE_TARGET_TEMP = (COMFORT_LOW + COMFORT_HIGH) / 2.0 # Target middle of band
RULE_MAX_DEVIATION_HIGH = 2.0 # Max degrees above target before HIGH cooling
RULE_MIN_DEVIATION_LOW = 0.5  # Min degrees above target for LOW cooling
RULE_OFF_THRESHOLD = COMFORT_LOW + 0.2 # Turn off cooling if temp drops below this

# --- Load Resources --- (Keep as before, skip RL agent)
print("Loading resources..."); f_cfg = ForecastConfig()
try:
    full_data = pd.read_csv(DATA_CSV, index_col='timestamp', parse_dates=True, low_memory=False); print(f"Loaded data: {DATA_CSV}")
    feature_scaler = joblib.load(FEATURE_SCALER_PATH); print("Feature scaler loaded.")
    target_scaler = joblib.load(TARGET_SCALER_PATH); print("Target scaler loaded.")
    num_input_features = len(feature_scaler.feature_names_in_)
    forecaster_model = LSTM_TCN_Forecaster(input_size=num_input_features, lstm_hidden_size=f_cfg.LSTM_HIDDEN_SIZE, lstm_layers=f_cfg.LSTM_LAYERS, tcn_channels=f_cfg.TCN_CHANNELS, tcn_kernel_size=f_cfg.TCN_KERNEL_SIZE, tcn_dropout=f_cfg.TCN_DROPOUT, output_size=f_cfg.OUTPUT_SIZE, pred_horizon=f_cfg.PREDICTION_HORIZON).to(f_cfg.DEVICE)
    forecaster_model.load_state_dict(torch.load(FORECAST_MODEL_PATH, map_location=f_cfg.DEVICE)); forecaster_model.eval()
    print(f"Forecaster model loaded: {FORECAST_MODEL_PATH}.")
    print("RL Agent NOT loaded - Using Enhanced Rule-Based Controller.")
except FileNotFoundError as e: print(f"ERROR loading file: {e}"); exit()
except Exception as e: print(f"ERROR loading resources: {e}"); import traceback; traceback.print_exc(); exit()

# --- Initialize Simulation State ---
print("\nInitializing simulation...")
current_data_index = START_OFFSET_STEPS
if current_data_index + f_cfg.SEQ_LENGTH + SIMULATION_STEPS >= len(full_data): SIMULATION_STEPS = len(full_data) - current_data_index - f_cfg.SEQ_LENGTH - 1; print(f"Adjusted steps: {SIMULATION_STEPS}")
current_zone_temps = np.random.uniform(COMFORT_LOW, COMFORT_HIGH, size=num_zones)
pso_swarm = PSOSwarm(n_particles=PSO_PARTICLES)
simulation_log = []; log_columns = [...] # Same columns as before, excluding RL_Action_Cont
mapped_discrete_action = 0

# --- Simulation Loop ---
print(f"\nStarting simulation for {SIMULATION_STEPS} steps (Enhanced Rule-Based + Enhanced PSO)...")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

for step in range(SIMULATION_STEPS):
    loop_start_time_str = full_data.index[current_data_index + f_cfg.SEQ_LENGTH - 1]
    # 1. Prepare data & 2. Run Forecaster (Keep as before)
    seq_start_idx = current_data_index; seq_end_idx = current_data_index + f_cfg.SEQ_LENGTH; current_data_point_idx = seq_end_idx - 1
    if seq_end_idx > len(full_data): break
    try: # Prepare forecast input
        raw_sequence_df=full_data.iloc[seq_start_idx:seq_end_idx]; feature_cols=feature_scaler.feature_names_in_
        sequence_features=raw_sequence_df[feature_cols]; scaled_features=feature_scaler.transform(sequence_features.values)
        features_tensor=torch.FloatTensor(scaled_features).unsqueeze(0).to(f_cfg.DEVICE)
    except Exception as e: print(f"ERROR prep forecast step {step}: {e}"); break
    with torch.no_grad(): forecast_scaled = forecaster_model(features_tensor).squeeze(0).cpu().numpy()

    # 3. Get Current State Variables
    current_data_row = full_data.iloc[current_data_point_idx]
    outdoor_temp = current_data_row['air_temperature']
    avg_zone_temp = np.mean(current_zone_temps)
    # Get 1-hour forecast (unscaled) for rule logic
    forecast_h1_unscaled = target_scaler.inverse_transform([[forecast_scaled[0]]])[0,0]

    # 4. <<< ENHANCED RULE-BASED CONTROLLER >>>
    avg_temp_deviation = avg_zone_temp - RULE_TARGET_TEMP
    predicted_temp_h1 = avg_zone_temp + (forecast_h1_unscaled - avg_zone_temp) # Simplistic trend estimate

    # Default to OFF
    total_cooling_power_needed = 0
    mapped_discrete_action = 0

    # If current temp is above target OR predicted to rise above target soon
    if avg_zone_temp > RULE_OFF_THRESHOLD or predicted_temp_h1 > COMFORT_HIGH:
        if avg_temp_deviation > RULE_MAX_DEVIATION_HIGH: # Significantly hot
            total_cooling_power_needed = COOLING_POWER_HIGH
            mapped_discrete_action = 3
        elif avg_temp_deviation > RULE_MIN_DEVIATION_LOW: # Moderately hot
            total_cooling_power_needed = COOLING_POWER_LOW
            mapped_discrete_action = 2
        # else: stay OFF if only slightly above target or only predicted rise is small

    # If current temp is below OFF threshold, ensure cooling is OFF
    if avg_zone_temp < RULE_OFF_THRESHOLD:
        total_cooling_power_needed = 0
        mapped_discrete_action = 0

    # (Could add FAN_ONLY logic here if desired, e.g., if temp is OK but needs circulation)

    # <<< END ENHANCED RULE-BASED CONTROLLER >>>

    # 5. Run PSO Controller (Uses enhanced fitness function from pso_controller.py)
    best_proportions = pso_swarm.optimize(current_temps=current_zone_temps, outdoor_temp=outdoor_temp, total_cooling_power=total_cooling_power_needed, comfort_low=COMFORT_LOW, comfort_high=COMFORT_HIGH, max_iter=PSO_MAX_ITER)

    # 6. Calculate Zone Watts
    zone_cooling_watts = best_proportions * total_cooling_power_needed

    # 7. Step Simulation
    next_zone_temps, delta_temps = simulate_multi_zone_step(current_temps=current_zone_temps, outdoor_temp=outdoor_temp, cooling_allocations=zone_cooling_watts)

    # 8. XAI Rule Check
    pre_step_rl_state_dict = {"current_temp_c": avg_zone_temp}; rule_explanations = check_basic_rules(pre_step_rl_state_dict, mapped_discrete_action)

    # 9. Log Data
    log_entry = {"Timestamp": loop_start_time_str, "Step": step + 1, "Outdoor_Temp": outdoor_temp}
    for i in range(num_zones): log_entry[f"Zone{i}_Temp_C"]=current_zone_temps[i]; log_entry[f"Zone{i}_Alloc_Prop"]=best_proportions[i]; log_entry[f"Zone{i}_Alloc_Watts"]=zone_cooling_watts[i]
    log_entry["Total_Cooling_Need_W"]=total_cooling_power_needed; log_entry["RL_Action_Disc"]=mapped_discrete_action; log_entry["XAI_Rule_Check"]="; ".join(rule_explanations)
    for h in range(f_cfg.PREDICTION_HORIZON): forecast_val_scaled=forecast_scaled[h]; forecast_val_unscaled=target_scaler.inverse_transform([[forecast_val_scaled]])[0,0]; log_entry[f"Forecast_h{h+1}"]=forecast_val_unscaled
    simulation_log.append(log_entry)

    # Print progress summary
    if (step + 1) % 24 == 0 or step == 0:
        print(f"\n--- Step {step+1}/{SIMULATION_STEPS} (Timestamp: {loop_start_time_str}) ---")
        print(f"  Temps Before: {[f'{t:.1f}' for t in current_zone_temps]}")
        print(f"  Rule Action -> Total Effort: {mapped_discrete_action} ({['OFF','FAN','LOW','HIGH'][mapped_discrete_action]}) -> {total_cooling_power_needed} W")
        print(f"  PSO Alloc Props: {[f'{p:.2f}' for p in best_proportions]}")
        print(f"  Temps After: {[f'{t:.1f}' for t in next_zone_temps]}")
        print(f"  XAI Rules: {rule_explanations}")

    # 10. Update State
    current_zone_temps = next_zone_temps
    current_data_index += 1

warnings.resetwarnings()
print("\nSimulation finished.")

# --- Save Log ---
if simulation_log:
    log_df = pd.DataFrame(simulation_log); log_df.to_csv(LOG_FILE, index=False); print(f"Simulation log saved to {LOG_FILE}")
else: print("No simulation steps logged.")