# hvac_env.py (NO FORECASTS, Continuous Actions, Quadratic Reward, Basic XAI)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib
import torch
import os
import random
import warnings

# Import the forecasting model class (Not strictly needed now, but keep for config)
try:
    from train_forecaster import LSTM_TCN_Forecaster, Config as ForecastConfig
except ImportError as e:
    # Dummy classes
    print("Warning: Could not import forecasting model components. Using dummy config.")
    class ForecastConfig: # Dummy Config
        SEQ_LENGTH = 24; PREDICTION_HORIZON = 6; DEVICE = torch.device("cpu"); INPUT_FEATURES = 60 # INPUT_FEATURES less critical now
        LSTM_HIDDEN_SIZE = 64; LSTM_LAYERS = 2; TCN_CHANNELS = [64, 128]; TCN_KERNEL_SIZE = 3
        TCN_DROPOUT = 0.2; OUTPUT_SIZE = 1; OUTPUT_DIR = './output/'; BUILDING_ID = 7; METER_TYPE = 1
        base_name = f"b{BUILDING_ID}_m{METER_TYPE}"
        original_featured_file = os.path.join(OUTPUT_DIR, f'featured_building_{base_name}.csv')
        model_save_path = os.path.join(OUTPUT_DIR, f'lstm_tcn_forecaster_{base_name}.pth')
        feature_scaler_file = os.path.join(OUTPUT_DIR, f'feature_scaler_{base_name}.joblib')
        target_scaler_file = os.path.join(OUTPUT_DIR, f'target_scaler_{base_name}.joblib')
    class LSTM_TCN_Forecaster(torch.nn.Module): # Dummy Model
        def __init__(self, *args, **kwargs): super().__init__(); self.fc = torch.nn.Linear(1,1); print("Using dummy LSTM_TCN_Forecaster.")
        def forward(self, x): print("Dummy forecaster called - shouldn't happen in no-forecast env"); return torch.rand(x.size(0), 6)


# --- Environment Constants --- (Keep as before)
HEAT_CAPACITY_AIR = 1005; AIR_DENSITY = 1.225; ROOM_VOLUME = 5*5*3; ROOM_MASS_AIR = ROOM_VOLUME*AIR_DENSITY
THERMAL_MASS_MULTIPLIER = 100.0; ROOM_THERMAL_MASS_CAPACITY = ROOM_MASS_AIR*HEAT_CAPACITY_AIR*THERMAL_MASS_MULTIPLIER
print(f"Adjusted ROOM_THERMAL_MASS_CAPACITY: {ROOM_THERMAL_MASS_CAPACITY:.2e} J/K")
WALL_AREA=(5*3)*4+(5*5); WALL_U_VALUE=1.5; WALL_R_VALUE=1/(WALL_U_VALUE*WALL_AREA)
COOLING_POWER_LOW = 3000; COOLING_POWER_HIGH = 6000; INTERNAL_GAIN_OCCUPIED = 500; INTERNAL_GAIN_UNOCCUPIED = 50
TIME_STEP = 3600; COMFORT_LOW = 21.0; COMFORT_HIGH = 24.0
ENERGY_COST_OFF = 0.0; ENERGY_COST_FAN_ONLY = 0.1; ENERGY_COST_LOW = 1.0; ENERGY_COST_HIGH = 2.0
CYCLING_PENALTY = 0.5; DISCOMFORT_PENALTY_FACTOR = 1.0
print(f"Using QUADRATIC discomfort penalty factor: {DISCOMFORT_PENALTY_FACTOR}")

# --- XAI Rule Checker Function --- (Keep as before)
def check_basic_rules(observation_dict, discrete_action_idx):
    # ... (keep implementation as before) ...
    explanations = []; temp_c = observation_dict.get("current_temp_c", None)
    action_name_map = {0:'OFF', 1:'FAN_ONLY', 2:'COOL_LOW', 3:'COOL_HIGH'}
    action_name = action_name_map.get(discrete_action_idx, "UNKNOWN")
    if temp_c is None or not isinstance(temp_c, (int, float)): return ["Cannot check rules: Temp missing/invalid."]
    if temp_c > COMFORT_HIGH + 0.5:
        if discrete_action_idx in [2, 3]: explanations.append(f"Rule: OK(Hot) - Temp({temp_c:.1f})>ComfortHigh, Cooling({action_name}).")
        else: explanations.append(f"Rule: WARN(Hot) - Temp({temp_c:.1f})>ComfortHigh, NO Cooling({action_name}).")
    elif temp_c < COMFORT_LOW - 0.5:
        if discrete_action_idx not in [2, 3]: explanations.append(f"Rule: OK(Cold) - Temp({temp_c:.1f})<ComfortLow, No Cooling({action_name}).")
        else: explanations.append(f"Rule: WARN(Cold) - Temp({temp_c:.1f})<ComfortLow, Cooling({action_name}).")
    elif COMFORT_LOW <= temp_c <= COMFORT_HIGH:
        if discrete_action_idx in [0, 1]: explanations.append(f"Rule: OK(Comfort) - Temp({temp_c:.1f}) in band, Minimal Action({action_name}).")
    return explanations

# --- Environment Class ---
class HVACEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, data_csv_path, forecast_model_path, feature_scaler_path, target_scaler_path, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        print("Initializing HVAC Environment (NO FORECASTS)...") # Indicate no forecasts
        try: # Load data/scalers
            print(f"Attempting to load data from: {data_csv_path}")
            self.data = pd.read_csv(data_csv_path, index_col='timestamp', parse_dates=True, low_memory=False)
            # Feature scaler only needed if time features aren't already scaled
            self.feature_scaler = joblib.load(feature_scaler_path) # Keep for now, might remove later if only temps used
            self.target_scaler = joblib.load(target_scaler_path)
            print("Data and scalers loaded.")
        except Exception as e: raise RuntimeError(f"Error loading env resources: {e}") from e

        # Forecaster NOT loaded or used in this version
        self.forecast_config = ForecastConfig() # Keep config for constants if needed

        # Action Space (Continuous Box)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        print(f"Action Space Type: {type(self.action_space)}")
        self._discrete_action_to_state_map = { 0: {'name': 'OFF', 'power': 0, 'energy': ENERGY_COST_OFF}, 1: {'name': 'FAN_ONLY', 'power': 0, 'energy': ENERGY_COST_FAN_ONLY}, 2: {'name': 'COOL_LOW', 'power': COOLING_POWER_LOW, 'energy': ENERGY_COST_LOW}, 3: {'name': 'COOL_HIGH', 'power': COOLING_POWER_HIGH, 'energy': ENERGY_COST_HIGH}, }
        self.num_discrete_actions = 4

        # <<< MODIFICATION: Define Observation Space WITHOUT Forecasts >>>
        # forecast_len = self.forecast_config.PREDICTION_HORIZON # REMOVED
        time_features = 4 # hour_sin/cos, dayofweek_sin/cos
        action_encoding = self.num_discrete_actions
        num_base_features = 3 # current temp, outdoor temp, dew temp
        num_comfort_features = 2 # low/high band

        # Calculate size WITHOUT forecast_len
        obs_size = num_base_features + time_features + action_encoding + num_comfort_features
        self.observation_size = obs_size
        print(f"Calculated observation space size (NO FORECASTS): {self.observation_size}")

        # Define bounds
        low_bounds=np.full(obs_size,-np.inf,dtype=np.float32); high_bounds=np.full(obs_size,np.inf,dtype=np.float32)
        current_idx=0
        low_bounds[current_idx:current_idx+num_base_features]=0.0; high_bounds[current_idx:current_idx+num_base_features]=1.0; current_idx+=num_base_features
        # Skip forecast index range
        low_bounds[current_idx:current_idx+time_features]=-1.0; high_bounds[current_idx:current_idx+time_features]=1.0; current_idx+=time_features
        low_bounds[current_idx:current_idx+action_encoding]=0.0; high_bounds[current_idx:current_idx+action_encoding]=1.0; current_idx+=action_encoding
        low_bounds[current_idx:current_idx+num_comfort_features]=0.0; high_bounds[current_idx:current_idx+num_comfort_features]=1.0
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        # <<< END MODIFICATION >>>

        # Env State Vars
        self.current_data_index = -1; self.current_temp_c = 22.0; self.current_discrete_action = 0; self.steps_taken = 0
        # Max steps calculation doesn't depend on forecast horizon directly here, but on SEQ_LENGTH for historical data access
        self.max_steps = len(self.data) - self.forecast_config.SEQ_LENGTH - 2 # Need at least SEQ_LENGTH history for features
        if self.max_steps <= 0: raise ValueError("Data too short."); print(f"Max steps per episode: {self.max_steps}")

    def _get_observation(self):
        # Need to access data point corresponding to the current state
        # Use the index *before* the one used for dynamics in step()
        current_obs_idx = self.current_data_index + self.forecast_config.SEQ_LENGTH - 1
        if current_obs_idx >= len(self.data): raise IndexError("Data index out of bounds for observation")

        # <<< MODIFICATION: Observation without forecast >>>
        current_data_row = self.data.iloc[current_obs_idx]
        required_cols = ['air_temperature','dew_temperature','hour_sin','hour_cos','dayofweek_sin','dayofweek_cos']
        if not all(col in current_data_row.index for col in required_cols): raise ValueError("Missing observation cols")
        outdoor_temp = current_data_row['air_temperature']; dew_temp = current_data_row['dew_temperature']
        hour_sin = current_data_row['hour_sin']; hour_cos = current_data_row['hour_cos']
        dayofweek_sin = current_data_row['dayofweek_sin']; dayofweek_cos = current_data_row['dayofweek_cos']

        try:
            current_temp_c_float=float(self.current_temp_c); outdoor_temp_float=float(outdoor_temp); dew_temp_float=float(dew_temp); comfort_low_float=float(COMFORT_LOW); comfort_high_float=float(COMFORT_HIGH)
            current_temp_scaled=self.target_scaler.transform([[current_temp_c_float]])[0,0]; outdoor_temp_scaled=self.target_scaler.transform([[outdoor_temp_float]])[0,0]; dew_temp_scaled=self.target_scaler.transform([[dew_temp_float]])[0,0]; comfort_low_scaled=self.target_scaler.transform([[comfort_low_float]])[0,0]; comfort_high_scaled=self.target_scaler.transform([[comfort_high_float]])[0,0]
        except Exception as e: print(f"Error scaling temperatures: {e}"); raise

        action_one_hot=np.zeros(self.num_discrete_actions,dtype=np.float32); action_one_hot[self.current_discrete_action]=1.0

        # --- Concatenate WITHOUT forecast ---
        arr1=np.array([current_temp_scaled,outdoor_temp_scaled,dew_temp_scaled],dtype=np.float32);
        # arr2 = forecast_scaled... # REMOVED
        arr3=np.array([hour_sin,hour_cos,dayofweek_sin,dayofweek_cos],dtype=np.float32);
        arr4=action_one_hot;
        arr5=np.array([comfort_low_scaled,comfort_high_scaled],dtype=np.float32)
        try: observation=np.concatenate([arr1,arr3,arr4,arr5]) # Concatenate remaining parts
        except ValueError as e: print(f"Concatenation failed! Shapes: {arr1.shape},{arr3.shape},{arr4.shape},{arr5.shape}"); raise e
        # <<< END MODIFICATION >>>

        if len(observation)!=self.observation_size: raise ValueError(f"Observation length mismatch...")
        return observation

    def _get_info(self): # Keep as before
        current_data_idx_for_info = self.current_data_index + self.forecast_config.SEQ_LENGTH - 1
        if current_data_idx_for_info >= len(self.data): return {"status": "Index out of bounds"}
        timestamp = self.data.index[current_data_idx_for_info]
        outdoor_temp_c = self.data.iloc[current_data_idx_for_info]['air_temperature']
        discrete_action_name = self._discrete_action_to_state_map[self.current_discrete_action]['name']
        info = {"current_temp_c": self.current_temp_c, "outdoor_temp_c": outdoor_temp_c, "current_action": discrete_action_name, "data_timestamp": timestamp}
        if hasattr(self, '_last_cont_action'): info["action_continuous"] = self._last_cont_action
        return info

    def reset(self, seed=None, options=None): # Keep as before
        super().reset(seed=seed); max_start_index = max(0, (len(self.data)//2) - self.forecast_config.SEQ_LENGTH - self.forecast_config.PREDICTION_HORIZON)
        self.current_data_index = self.np_random.integers(0, max_start_index + 1); self.current_temp_c = self.np_random.uniform(COMFORT_LOW, COMFORT_HIGH)
        self.current_discrete_action = 0; self.steps_taken = 0; self._last_cont_action = None
        try: observation = self._get_observation(); info = self._get_info(); print(f"Env reset (NO FORECAST). Idx:{self.current_data_index}, Temp:{self.current_temp_c:.1f}C") # Modified print
        except Exception as e: print(f"Error in reset _get_observation: {e}"); observation = np.zeros(self.observation_space.shape, dtype=np.float32); info = {"error":"Reset failed"}
        if self.render_mode == "human": self._render_human(info, reward=0)
        return observation, info

    def step(self, action_cont): # Keep quadratic reward logic, XAI check etc.
        # --- Map continuous action & Apply action costs ---
        if not self.action_space.contains(action_cont): action_cont = np.clip(action_cont, self.action_space.low, self.action_space.high)
        action_val = action_cont[0]; self._last_cont_action = action_val
        if action_val < 0.25: mapped_discrete_action = 0
        elif action_val < 0.5: mapped_discrete_action = 1
        elif action_val < 0.75: mapped_discrete_action = 2
        else: mapped_discrete_action = 3
        previous_discrete_action = self.current_discrete_action
        self.current_discrete_action = mapped_discrete_action
        hvac_state = self._discrete_action_to_state_map[self.current_discrete_action]
        cooling_power = hvac_state['power']; energy_cost = hvac_state['energy']
        cycling_cost = 0;
        if (self.current_discrete_action in [2,3]) != (previous_discrete_action in [2,3]): cycling_cost = CYCLING_PENALTY

        # --- Get conditions & Simulate temp ---
        current_data_idx_for_dynamics = self.current_data_index + self.forecast_config.SEQ_LENGTH - 1
        if current_data_idx_for_dynamics >= len(self.data): return np.zeros(self.observation_space.shape,dtype=np.float32),0,False,True,{"status":"Truncated"}
        current_data_row=self.data.iloc[current_data_idx_for_dynamics]; outdoor_temp_c=current_data_row['air_temperature']; timestamp=current_data_row.name
        is_occupied = 9 <= timestamp.hour < 17 and timestamp.dayofweek < 5; internal_gain = INTERNAL_GAIN_OCCUPIED if is_occupied else INTERNAL_GAIN_UNOCCUPIED
        heat_flow_external = (outdoor_temp_c - self.current_temp_c) / WALL_R_VALUE
        net_heat_flow = heat_flow_external + internal_gain - cooling_power
        delta_temp = (net_heat_flow * TIME_STEP) / ROOM_THERMAL_MASS_CAPACITY
        temp_before_step = self.current_temp_c
        self.current_temp_c += delta_temp

        # --- Calculate QUADRATIC discomfort cost ---
        discomfort_cost = 0
        if self.current_temp_c < COMFORT_LOW: discomfort_cost = DISCOMFORT_PENALTY_FACTOR * (COMFORT_LOW - self.current_temp_c)**2
        elif self.current_temp_c > COMFORT_HIGH: discomfort_cost = DISCOMFORT_PENALTY_FACTOR * (self.current_temp_c - COMFORT_HIGH)**2

        # --- Calculate Reward ---
        reward = - (energy_cost + discomfort_cost + cycling_cost)

        # --- Update state & check termination ---
        self.current_data_index += 1; self.steps_taken += 1
        terminated = False; truncated = self.steps_taken >= self.max_steps or self.current_data_index >= (len(self.data) - self.forecast_config.SEQ_LENGTH - 1) # Adjusted truncation check

        # --- Get next observation & info (including XAI) ---
        try: observation = self._get_observation()
        except IndexError: truncated=True; observation=np.zeros(self.observation_space.shape,dtype=np.float32); print(f"IndexError getting obs in step {self.steps_taken}")
        except Exception as e: truncated=True; observation=np.zeros(self.observation_space.shape,dtype=np.float32); print(f"Error getting obs in step: {e}")

        info = self._get_info()
        pre_step_state_dict = {"current_temp_c": temp_before_step}
        rule_explanations = check_basic_rules(pre_step_state_dict, self.current_discrete_action)
        info["xai_rules"] = rule_explanations
        info["reward"]=reward; info["energy_cost"]=-energy_cost; info["discomfort_cost"]=-discomfort_cost; info["cycling_cost"]=-cycling_cost; info["delta_temp"]=delta_temp; info["action_continuous"]=self._last_cont_action

        if self.render_mode == "human": self._render_human(info, reward)
        if not np.isfinite(reward): reward = -1e6; truncated = True

        return observation, reward, terminated, truncated, info

    # --- render, _render_human, close --- (Keep as before)
    def render(self): # ... (keep implementation) ...
        if self.render_mode=='ansi': info=self._get_info(); return f"T:{info.get('ts','N/A')},Temp:{info.get('temp','N/A'):.1f},Out:{info.get('out','N/A'):.1f},Act:{info.get('act','N/A')}"
    def _render_human(self, info, reward): # ... (keep implementation) ...
        def format_or_na(v,f): return f"{v:{f}}" if isinstance(v,(int,float)) and np.isfinite(v) else str(v) if v is not None else 'N/A'
        dt=info.get('delta_temp','N/A'); ca=info.get('action_continuous','N/A'); rules=info.get('xai_rules',[]); rule_str=" | Rules: "+", ".join(rules) if rules else ""
        print(f"T:{info.get('data_timestamp','N/A')} | Tmp:{format_or_na(info.get('current_temp_c'),'.1f')}°C(ΔT:{format_or_na(dt,'.2f')}) | Out:{format_or_na(info.get('outdoor_temp_c'),'.1f')}°C | Act:{str(info.get('current_action','N/A')):<9}(Cont:{format_or_na(ca,'.2f')}) | Rew:{format_or_na(reward,'<7.2f')} [E:{format_or_na(info.get('energy_cost',0),'<4.1f')} D:{format_or_na(info.get('discomfort_cost',0),'<6.1f')} C:{format_or_na(info.get('cycling_cost',0),'<3.1f')}]{rule_str}")
    def close(self): print("Closing HVAC Environment.")

# --- Example Usage ---
if __name__ == '__main__':
    print("Testing HVACEnv (NO FORECASTS, Cont Action, Quadratic Reward, XAI Rules)...")
    try: cfg_test=ForecastConfig(); env_config={"data_csv_path": cfg_test.original_featured_file,"forecast_model_path": cfg_test.model_save_path,"feature_scaler_path": cfg_test.feature_scaler_file,"target_scaler_path": cfg_test.target_scaler_file,"render_mode": "human"}; print("Using paths from ForecastConfig.")
    except Exception: print("Warning: Using explicit paths."); output_dir='./output/'; base_name="b7_m1"; env_config={"data_csv_path":os.path.join(output_dir,f'featured_building_{base_name}.csv'),"forecast_model_path":os.path.join(output_dir,f'lstm_tcn_forecaster_{base_name}.pth'),"feature_scaler_path":os.path.join(output_dir,f'feature_scaler_{base_name}.joblib'),"target_scaler_path":os.path.join(output_dir,f'target_scaler_{base_name}.joblib'),"render_mode":"human"}
    print(f"Env config: {env_config}")
    print(f"Attempting to load data_csv_path: {env_config['data_csv_path']}")
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        env = HVACEnv(**env_config);
        # Test observation space directly after init
        print("Observation Space Sample:", env.observation_space.sample())
        print("Observation Space Shape:", env.observation_space.shape)

        obs, info = env.reset(); print("Reset successful."); print("Action Space:", env.action_space); print("Initial Obs Shape:", obs.shape); print("Initial Info:", info)
        print("\nTesting step function..."); total_reward=0; n_steps_test=5 # Just a few steps
        for i in range(n_steps_test):
            action=env.action_space.sample(); print(f"\n--- Step {i+1}/{n_steps_test} ---")
            pre_step_info=env._get_info(); print(f"State before: T={pre_step_info.get('current_temp_c','N/A'):.1f} Act={pre_step_info.get('current_action','N/A')}")
            print(f"Sampled Cont Action: {action[0]:.3f}")
            obs, reward, terminated, truncated, info = env.step(action); total_reward += reward; print(f"Returned Obs Shape: {obs.shape}"); print(f"Terminated: {terminated}, Truncated: {truncated}")
            if terminated or truncated: print("Episode ended."); break
        warnings.resetwarnings(); print(f"\nTotal reward over {i+1} random steps: {total_reward:.2f}"); env.close(); print("\nEnvironment test finished.")
    except Exception as e: print(f"\nERROR during testing: {e}"); import traceback; traceback.print_exc()