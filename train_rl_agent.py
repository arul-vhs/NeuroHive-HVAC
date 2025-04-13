# train_rl_agent.py (Using SAC with Simple Reward Env)

import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC # Use SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import warnings
import numpy as np

# Import the custom environment (MUST be the Simple Reward version)
try:
    from hvac_env import HVACEnv, ForecastConfig
except ImportError:
    print("Error importing HVACEnv. Ensure hvac_env.py is present and correct.")
    exit()

# --- Configuration ---
TOTAL_TIMESTEPS = 500000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 5
MODEL_ALGO = "SAC"
NUM_CPU = 4
LEARNING_RATE = 1e-4
SAC_BATCH_SIZE = 1024
# <<< Updated log suffix >>>
TB_LOG_SUFFIX = f"SAC_lr1e-4_n{NUM_CPU}_b{SAC_BATCH_SIZE}_contAct_SimpleRwd"

# --- Paths --- (Keep as before)
try: cfg_paths = ForecastConfig(); DATA_CSV=cfg_paths.original_featured_file; FORECAST_MODEL=cfg_paths.model_save_path; FEATURE_SCALER=cfg_paths.feature_scaler_file; TARGET_SCALER=cfg_paths.target_scaler_file; BUILDING_ID=cfg_paths.BUILDING_ID; METER_TYPE=cfg_paths.METER_TYPE
except Exception: print("Warning: Using explicit paths."); output_dir='./output/'; base_name="b7_m1"; BUILDING_ID=7; METER_TYPE=1; DATA_CSV=os.path.join(output_dir,f'featured_building_{base_name}.csv'); FORECAST_MODEL=os.path.join(output_dir,f'lstm_tcn_forecaster_{base_name}.pth'); FEATURE_SCALER=os.path.join(output_dir,f'feature_scaler_{base_name}.joblib'); TARGET_SCALER=os.path.join(output_dir,f'target_scaler_{base_name}.joblib')

# --- Directories --- (Update names)
LOG_DIR="./rl_logs/"; TENSORBOARD_LOG_DIR="./rl_tensorboard_logs/"
BEST_MODEL_SAVE_PATH = os.path.join(LOG_DIR, f"best_model_{TB_LOG_SUFFIX}_b{BUILDING_ID}m{METER_TYPE}")
FINAL_MODEL_SAVE_NAME = f"final_model_{TB_LOG_SUFFIX}_b{BUILDING_ID}m{METER_TYPE}"
FINAL_MODEL_SAVE_PATH = os.path.join(LOG_DIR, FINAL_MODEL_SAVE_NAME)
os.makedirs(LOG_DIR, exist_ok=True); os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True); os.makedirs(BEST_MODEL_SAVE_PATH, exist_ok=True)

# --- Environment Kwargs --- (Keep as before)
env_kwargs = {"data_csv_path":DATA_CSV, "forecast_model_path":FORECAST_MODEL, "feature_scaler_path":FEATURE_SCALER, "target_scaler_path":TARGET_SCALER, "render_mode":None}

# --- make_env function --- (Update monitor filename)
def make_env(rank: int, seed: int = 0, env_options: dict = None):
    if env_options is None: env_options = {}
    def _init():
        env = HVACEnv(**env_options)
        monitor_filename = os.path.join(LOG_DIR, f'monitor_{TB_LOG_SUFFIX}_{rank}.csv') if LOG_DIR else None
        env = Monitor(env, filename=monitor_filename)
        return env
    return _init

# --- Main Execution Guard ---
if __name__ == '__main__':
    # --- Environment Setup --- (Keep as before)
    print("Setting up environment..."); env_seed=42
    try:
        print(f"Creating {NUM_CPU} parallel environments..."); env = SubprocVecEnv([make_env(i,env_seed,env_kwargs) for i in range(NUM_CPU)]); env.seed(env_seed)
        eval_env = DummyVecEnv([make_env(0, env_seed + NUM_CPU, env_kwargs)]); eval_env.seed(env_seed + NUM_CPU)
        print("Environments created."); print(f"Action Space: {env.action_space}"); print(f"Observation Space: {env.observation_space}")
    except Exception as e: print(f"\nERROR env setup: {e}"); import traceback; traceback.print_exc(); exit()

    # --- Callbacks Setup --- (Keep as before)
    print("Setting up callbacks..."); eval_freq_per_env = max(EVAL_FREQ//NUM_CPU, 1); print(f"Eval Freq per env: {eval_freq_per_env}")
    eval_callback = EvalCallback(eval_env, best_model_save_path=BEST_MODEL_SAVE_PATH, log_path=LOG_DIR, eval_freq=eval_freq_per_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True, render=False)

    # --- Model Training --- (Keep SAC init)
    print(f"Initializing {MODEL_ALGO} model..."); model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOG_DIR, learning_rate=LEARNING_RATE, buffer_size=100000, batch_size=SAC_BATCH_SIZE, gamma=0.99, tau=0.005, train_freq=1, gradient_steps=1, learning_starts=10000, use_sde=False, device="auto")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}"); print(f"Algorithm: {MODEL_ALGO}"); print(f"LR: {LEARNING_RATE}"); print(f"Batch Size: {SAC_BATCH_SIZE}")

    print(f"Starting training ({MODEL_ALGO} with Simple Reward)...")
    try:
        # <<< Use Updated Log Name >>>
        tb_log_name = f"{MODEL_ALGO}_b{BUILDING_ID}m{METER_TYPE}_{TB_LOG_SUFFIX}"
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, tb_log_name=tb_log_name, log_interval=1)
        print("Training finished.")
        print(f"Saving final model to: {FINAL_MODEL_SAVE_PATH}")
        model.save(FINAL_MODEL_SAVE_PATH)

        # --- Post-Training Evaluation --- (Keep as before, using SAC.load)
        print("\nLoading best model found during training...")
        best_model_path = os.path.join(BEST_MODEL_SAVE_PATH, "best_model")
        if os.path.exists(best_model_path + ".zip"):
            best_model = SAC.load(best_model_path, env=eval_env) # Load SAC
            print("Evaluating the best model..."); obs, _ = eval_env.reset()
            n_test_episodes = 3; episodes_done = 0
            with warnings.catch_warnings():
                 warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                 while episodes_done < n_test_episodes:
                      action, _ = best_model.predict(obs, deterministic=True)
                      obs, reward, terminated, truncated, info = eval_env.step(action)
                      if terminated[0] or truncated[0]:
                           episodes_done += 1
                           if 'episode' in info[0]: print(f"Test Ep {episodes_done} Reward: {info[0]['episode']['r']:.2f}, Length: {info[0]['episode']['l']}")
                           else: print(f"Test Ep {episodes_done} finished.")
            print(f"\nFinished evaluation.")
        else: print(f"Could not find best model at {best_model_path}.zip.")

    except Exception as e: print(...); import traceback; traceback.print_exc()
    finally: # --- Cleanup --- (Keep as before)
        if 'env' in locals() and env is not None: env.close()
        if 'eval_env' in locals() and eval_env is not None: eval_env.close()
        print("\nEnvironments closed.")

    print("\nScript finished.")
    print(f"To view training logs, run: tensorboard --logdir {TENSORBOARD_LOG_DIR}")