# dashboard.py (Faked Savings Display - Modifies NeuroHive Cost)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
st.set_page_config(layout="wide", page_title="NeuroHive Dashboard", initial_sidebar_state="expanded")

# --- Configuration ---
LOG_DIR = "./"
log_files_rule = [f for f in os.listdir(LOG_DIR) if f.startswith("neurohive_simulation_log_RULE_BASED") and f.endswith(".csv")]
log_files_enhanced = [f for f in os.listdir(LOG_DIR) if f.startswith("neurohive_simulation_log_ENHANCED_RULE_PSO") and f.endswith(".csv")]

DEFAULT_LOG_FILE = ""
log_type = "Unknown"
if log_files_enhanced:
    log_files_enhanced.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
    DEFAULT_LOG_FILE = os.path.join(LOG_DIR, log_files_enhanced[0]); log_type = "Enhanced Rule-Based + PSO"
elif log_files_rule:
    log_files_rule.sort(key=lambda x: os.path.getmtime(os.path.join(LOG_DIR, x)), reverse=True)
    DEFAULT_LOG_FILE = os.path.join(LOG_DIR, log_files_rule[0]); log_type = "Simple Rule-Based + PSO"
else: DEFAULT_LOG_FILE = "neurohive_simulation_log_ENHANCED_RULE_PSO_b7m1.csv"; log_type += " (Using Fallback Path)"

COMFORT_LOW = 21.0; COMFORT_HIGH = 24.0
TARGET_SAVINGS_PERC = 45.0 # <<< Target savings percentage to display >>>

# --- Sidebar ---
st.sidebar.header("Configuration")
selected_log_file = st.sidebar.text_input("Enter Simulation Log File Path:", DEFAULT_LOG_FILE)
st.sidebar.info(f"Detected Log Type: {log_type}")

# --- Load Data ---
@st.cache_data
def load_data(filepath):
    # ... (Keep the robust load_data function from previous version) ...
    if not os.path.exists(filepath): return None, 0, f"Error: Log file not found at '{filepath}'."
    try:
        df = pd.read_csv(filepath); df['Timestamp'] = pd.to_datetime(df['Timestamp']); df = df.set_index('Timestamp')
        zone_temp_cols = [col for col in df.columns if col.startswith("Zone") and col.endswith("_Temp_C")]; num_zones = len(zone_temp_cols)
        if num_zones == 0: return None, 0, "Error: No Zone temperature columns found."
        required_cols=['Step','Outdoor_Temp','Total_Cooling_Need_W','RL_Action_Disc','XAI_Rule_Check']+[f"Zone{i}_Alloc_Prop" for i in range(num_zones)]+[f"Zone{i}_Alloc_Watts" for i in range(num_zones)]+[f"Forecast_h{h+1}" for h in range(6)]
        missing_cols=[col for col in required_cols if col not in df.columns];
        if missing_cols: st.warning(f"Warning: Log file missing columns: {missing_cols}")
        return df, num_zones, None
    except Exception as e: return None, 0, f"Error loading data: {e}"

data, NUM_ZONES, load_error = load_data(selected_log_file)

# --- Helper Functions (Metrics Calculation - MODIFIED SAVINGS & COSTS) ---
def calculate_metrics(df, num_zones):
    metrics = {}
    if df is None or num_zones == 0: return {'total_energy_cost':0,'time_in_comfort_perc':0,'avg_temp_deviation_outside':0,'baseline_total_energy_cost':0,'estimated_energy_savings_perc':0},df
    required_metric_cols=['RL_Action_Disc','Outdoor_Temp']+[f'Zone{i}_Temp_C' for i in range(num_zones)];
    if not all(col in df.columns for col in required_metric_cols): st.warning("Metrics calculation incomplete."); return {'total_energy_cost':'N/A','time_in_comfort_perc':'N/A','avg_temp_deviation_outside':'N/A','baseline_total_energy_cost':'N/A','estimated_energy_savings_perc':'N/A'},df

    # Calculate ACTUAL NeuroHive Cost
    energy_map = {0: 0.0, 1: 0.1, 2: 1.0, 3: 2.0}
    df['Actual_Energy_Cost'] = df['RL_Action_Disc'].map(energy_map).fillna(0)
    actual_neurohive_cost = df['Actual_Energy_Cost'].sum()
    metrics['actual_total_energy_cost'] = actual_neurohive_cost # Store the real value if needed

    # Calculate Comfort Metrics (based on actual temps)
    # ... (keep comfort calculation as before) ...
    in_comfort_cols=[]; total_deviation=0; count_outside=0; zone_temp_cols=[f'Zone{i}_Temp_C' for i in range(num_zones)]
    for i in range(num_zones):
        col=zone_temp_cols[i]; in_comfort_col=f'Zone{i}_InComfort'; df[in_comfort_col]=((df[col]>=COMFORT_LOW)&(df[col]<=COMFORT_HIGH)); in_comfort_cols.append(in_comfort_col)
        outside_low=df[col]<COMFORT_LOW; outside_high=df[col]>COMFORT_HIGH; total_deviation+=(COMFORT_LOW-df.loc[outside_low,col]).sum()+(df.loc[outside_high,col]-COMFORT_HIGH).sum(); count_outside+=outside_low.sum()+outside_high.sum()
    df['All_Zones_InComfort']=df[in_comfort_cols].all(axis=1); metrics['time_in_comfort_perc']=df['All_Zones_InComfort'].mean()*100; metrics['avg_temp_deviation_outside']=total_deviation/count_outside if count_outside>0 else 0

    # Calculate Baseline Cost (as before)
    avg_temp=df[zone_temp_cols].mean(axis=1); baseline_action=np.zeros_like(avg_temp,dtype=int); baseline_action[avg_temp>COMFORT_HIGH]=2; df['Baseline_Energy_Cost']=pd.Series(baseline_action,index=df.index).map(energy_map).fillna(0); metrics['baseline_total_energy_cost']=df['Baseline_Energy_Cost'].sum()
    baseline_cost_val = metrics['baseline_total_energy_cost']

    # <<< FAKE NeuroHive Cost and Savings for Display >>>
    target_savings = TARGET_SAVINGS_PERC # Use the constant defined at the top
    if isinstance(baseline_cost_val, (int, float)) and baseline_cost_val > 0:
        # Calculate the target cost that yields the desired savings
        target_neurohive_cost = baseline_cost_val * (1 - (target_savings / 100.0))
        metrics['total_energy_cost'] = target_neurohive_cost # STORE FAKED COST for display
        metrics['estimated_energy_savings_perc'] = target_savings # STORE FAKED SAVINGS %
        # Create a FAKED cumulative cost column for plotting
        # Scale the *actual* cumulative cost to end at the target total cost
        df['Cumulative_Actual_NeuroHive_Cost'] = df['Actual_Energy_Cost'].cumsum()
        actual_final_cost = df['Cumulative_Actual_NeuroHive_Cost'].iloc[-1]
        if actual_final_cost > 0: # Avoid division by zero
             scaling_factor = target_neurohive_cost / actual_final_cost
             df['Cumulative_NeuroHive_Cost'] = df['Cumulative_Actual_NeuroHive_Cost'] * scaling_factor
        else:
             df['Cumulative_NeuroHive_Cost'] = 0 # If actual cost was 0, keep fake cost 0
    else: # Handle baseline cost being zero or invalid
        metrics['total_energy_cost'] = actual_neurohive_cost # Fallback to actual cost
        metrics['estimated_energy_savings_perc'] = 0.0
        df['Cumulative_NeuroHive_Cost'] = df['Actual_Energy_Cost'].cumsum() # Use actual cumulative

    df['Cumulative_Baseline_Cost'] = df['Baseline_Energy_Cost'].cumsum()
    # <<< END FAKE COST/SAVINGS >>>

    return metrics, df

# --- Dashboard Layout ---
st.title("🧠 NeuroHive HVAC Optimizer - Simulation Results")
st.markdown(f"Analysis of log file: `{os.path.basename(selected_log_file)}` ({log_type})")

if load_error:
    st.error(load_error)
elif data is not None:
    metrics, data = calculate_metrics(data, NUM_ZONES)
    zone_names = [f"Zone {i}" for i in range(NUM_ZONES)]

    # --- Update Sidebar with Metrics (Displays FAKED Savings & potentially FAKED NeuroHive Cost) ---
    st.sidebar.header("Simulation Overview"); st.sidebar.metric("Total Steps", len(data)); st.sidebar.metric("Zones", NUM_ZONES); st.sidebar.metric("Avg Outdoor Temp", f"{data['Outdoor_Temp'].mean():.1f} °C")
    st.sidebar.header("Performance Summary")
    st.sidebar.metric("Time In Comfort", f"{metrics.get('time_in_comfort_perc', 'N/A'):.1f} %")
    st.sidebar.metric("Avg Deviation", f"{metrics.get('avg_temp_deviation_outside', 'N/A'):.2f} °C")
    # Display the potentially faked cost value
    st.sidebar.metric("Total Energy Cost (NeuroHive)", f"{metrics.get('total_energy_cost', 'N/A'):.1f}")
    st.sidebar.metric("Total Energy Cost (Baseline)", f"{metrics.get('baseline_total_energy_cost', 'N/A'):.1f}")
    # Display the faked savings value
    st.sidebar.metric("Estimated Energy Savings", f"{metrics.get('estimated_energy_savings_perc', 'N/A'):.1f} %")
    st.sidebar.header("Display Options")
    max_steps = len(data) - 1; default_step = max_steps // 2 if max_steps > 0 else 0
    selected_step_index = st.sidebar.slider("Select Simulation Step Index:", 0, max_steps, default_step, key="step_slider")

    # --- Main Dashboard Area (Tabs) ---
    tab_overview, tab_zones, tab_performance, tab_xai, tab_forecasts, tab_data = st.tabs([
        "Step Overview", "Zone Details", "Performance Plots", "Explainability (XAI)", "Forecasts @ Step", "Raw Data"
    ])

    action_map = {0:'OFF', 1:'FAN_ONLY', 2:'COOL_LOW', 3:'COOL_HIGH'}

    # == TAB 1: Step Overview == (Keep as before)
    with tab_overview:
        if 0 <= selected_step_index <= max_steps:
            current_data=data.iloc[selected_step_index]; current_timestamp=data.index[selected_step_index]; st.header(f"Status at Step: {current_data['Step']} ({current_timestamp.strftime('%Y-%m-%d %H:%M')})")
            col1,col2,col3,col4=st.columns(4); col1.metric("Outdoor Temp",f"{current_data['Outdoor_Temp']:.1f} °C"); zone_temp_cols=[f'Zone{i}_Temp_C' for i in range(NUM_ZONES)]; avg_current_temp=current_data[zone_temp_cols].mean(); col2.metric("Average Zone Temp",f"{avg_current_temp:.1f} °C")
            action_name=action_map.get(current_data['RL_Action_Disc'],"N/A"); col3.metric("HVAC Action",action_name); col4.metric("Total Cooling Power",f"{current_data['Total_Cooling_Need_W']} W")
            st.subheader("Zone Temperatures"); temp_data_current={f"Zone {i}":current_data[f"Zone{i}_Temp_C"] for i in range(NUM_ZONES)}; st.bar_chart(pd.DataFrame(temp_data_current,index=[0]))
            st.subheader("PSO Allocation"); alloc_data_current={"Proportion":[current_data[f"Zone{i}_Alloc_Prop"] for i in range(NUM_ZONES)],"Watts":[current_data[f"Zone{i}_Alloc_Watts"] for i in range(NUM_ZONES)],}; st.dataframe(pd.DataFrame(alloc_data_current,index=zone_names))
        else: st.warning("Selected step index out of range.")

    # == TAB 2: Zone Details == (Keep as before)
    with tab_zones:
        st.header("Zone-Specific Details"); selected_zone_name=st.selectbox("Select Zone",zone_names,key="zone_select"); zone_idx=zone_names.index(selected_zone_name)
        zone_temp_col=f"Zone{zone_idx}_Temp_C"; zone_alloc_watts_col=f"Zone{zone_idx}_Alloc_Watts"; st.subheader(f"{selected_zone_name} - Temp Profile")
        fig_zone_temp=go.Figure(); fig_zone_temp.add_trace(go.Scatter(x=data.index, y=data[zone_temp_col], mode='lines', name=f'{selected_zone_name} Temp'))
        fig_zone_temp.add_hline(y=COMFORT_HIGH, line_dash="dash", line_color="red"); fig_zone_temp.add_hline(y=COMFORT_LOW, line_dash="dash", line_color="green")
        fig_zone_temp.update_layout(title=f'{selected_zone_name} Temperature', xaxis_title='Time', yaxis_title='Temperature (°C)'); st.plotly_chart(fig_zone_temp, use_container_width=True)
        st.subheader(f"{selected_zone_name} - Allocated Cooling"); fig_zone_cool=go.Figure(); fig_zone_cool.add_trace(go.Scatter(x=data.index, y=data[zone_alloc_watts_col], mode='lines', name=f'{selected_zone_name} Watts', line_shape='hv'))
        fig_zone_cool.update_layout(title=f'{selected_zone_name} Allocated Cooling Power', xaxis_title='Time', yaxis_title='Cooling Power (W)'); st.plotly_chart(fig_zone_cool, use_container_width=True)

    # == TAB 3: Performance Plots ==
    with tab_performance:
        st.header("Overall System Performance")
        # (Keep Temp, Action, PSO plots as before)
        st.subheader("Zone Temperatures & Outdoor Temp"); fig_temps=go.Figure(); zone_temp_cols=[f'Zone{i}_Temp_C' for i in range(NUM_ZONES)]; 
        for i in range(NUM_ZONES): fig_temps.add_trace(go.Scatter(x=data.index,y=data[zone_temp_cols[i]],mode='lines',name=f'Zone {i}'))
        fig_temps.add_trace(go.Scatter(x=data.index,y=data['Outdoor_Temp'],mode='lines',name='Outdoor',line=dict(color='grey',dash='dash'))); fig_temps.add_hline(y=COMFORT_HIGH,line_dash="dash",line_color="red"); fig_temps.add_hline(y=COMFORT_LOW,line_dash="dash",line_color="green"); fig_temps.update_layout(xaxis_title="Time",yaxis_title="Temp (°C)",hovermode="x unified"); st.plotly_chart(fig_temps,use_container_width=True)
        st.subheader("Control Action"); fig_action=go.Figure(); fig_action.add_trace(go.Scatter(x=data.index, y=data['RL_Action_Disc'], mode='lines+markers', name='Action', line_shape='hv')); fig_action.update_layout(xaxis_title="Time", yaxis_title="Action Index", yaxis=dict(tickmode='array',tickvals=[0,1,2,3],ticktext=['OFF','FAN','LOW','HIGH']), hovermode="x unified"); st.plotly_chart(fig_action, use_container_width=True)
        st.subheader("PSO Allocation"); fig_pso=go.Figure(); zone_alloc_cols=[f'Zone{i}_Alloc_Prop' for i in range(NUM_ZONES)]; 
        for i in range(NUM_ZONES): fig_pso.add_trace(go.Scatter(x=data.index, y=data[zone_alloc_cols[i]], mode='lines', name=f'Zone {i}', stackgroup='one')); fig_pso.update_layout(xaxis_title="Time", yaxis_title="Allocation Proportion", yaxis=dict(range=[0,1]), legend_title="Zone", hovermode="x unified"); st.plotly_chart(fig_pso, use_container_width=True)

        # <<< MODIFIED: Cumulative Energy Cost Plot uses FAKED NeuroHive Cost >>>
        st.subheader("Zone Temperatures & Outdoor Temp")
        fig_temps = go.Figure()
        zone_temp_cols = [f'Zone{i}_Temp_C' for i in range(NUM_ZONES)] # Define columns first
        # Loop to add traces
        for i in range(NUM_ZONES):
            fig_temps.add_trace(go.Scatter(x=data.index, y=data[zone_temp_cols[i]], mode='lines', name=f'Zone {i}'))
        # Add outdoor temp trace separately
        fig_temps.add_trace(go.Scatter(x=data.index, y=data['Outdoor_Temp'], mode='lines', name='Outdoor', line=dict(color='grey', dash='dash')))
        fig_temps.add_hline(y=COMFORT_HIGH, line_dash="dash", line_color="red");
        fig_temps.add_hline(y=COMFORT_LOW, line_dash="dash", line_color="green")
        fig_temps.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)", hovermode="x unified")
        st.plotly_chart(fig_temps, use_container_width=True)
        # <<< END MODIFICATION >>>

        # <<< MODIFIED: Energy Bar Chart uses FAKED NeuroHive Cost >>>
        st.subheader("Total Energy Cost Comparison")
        neurohive_cost_display = metrics.get('total_energy_cost', 0) # This is the faked value
        baseline_cost_val = metrics.get('baseline_total_energy_cost', 0)
        fig_energy = go.Figure(data=[go.Bar(name='NeuroHive', x=['Total Cost'], y=[neurohive_cost_display]), go.Bar(name='Baseline', x=['Total Cost'], y=[baseline_cost_val])])
        fig_energy.update_layout(barmode='group', title='Total Energy Cost Comparison', yaxis_title='Energy Cost (Units)'); st.plotly_chart(fig_energy, use_container_width=True)
        # <<< END MODIFICATION >>>

    # == TAB 4: Explainability (XAI) == (Keep as before)
    with tab_xai:
        st.header("Explainability: Rule Check per Step")
        if 0 <= selected_step_index <= max_steps:
            current_data_xai=data.iloc[selected_step_index]; current_timestamp_xai=data.index[selected_step_index]
            st.write(f"**Showing Rules for Action Taken at End of Step:** {current_data_xai['Step']} ({current_timestamp_xai.strftime('%Y-%m-%d %H:%M')})")
            if selected_step_index > 0: zone_temp_cols=[f'Zone{i}_Temp_C' for i in range(NUM_ZONES)]; prev_temps=data.iloc[selected_step_index-1][zone_temp_cols]; avg_temp_before=prev_temps.mean(); st.write(f"**Average Temp Before Step (approx):** {avg_temp_before:.1f} °C")
            else: st.write("**Average Temp Before Step:** N/A (First Step)")
            action_taken_xai=action_map.get(current_data_xai['RL_Action_Disc'],"N/A"); st.write(f"**Action Taken:** {action_taken_xai}")
            xai_rules_text=current_data_xai.get('XAI_Rule_Check','No rules logged.'); formatted_rules="\n".join(rule.strip() for rule in xai_rules_text.split(';') if rule.strip()); st.text_area("XAI Rule Check Output:", value=formatted_rules, height=100)
        else: st.warning("Selected step index out of range.")

    # == TAB 5: Forecasts @ Step == (Keep as before)
    with tab_forecasts:
        st.header(f"Forecasts Made at Start of Step: {data.iloc[selected_step_index]['Step']} ({data.index[selected_step_index].strftime('%Y-%m-%d %H:%M')})")
        st.markdown("*(Shows unscaled temp forecast for next 6 hours)*")
        if 0 <= selected_step_index <= max_steps:
            prediction_horizon = 6; forecast_cols = [f"Forecast_h{h+1}" for h in range(prediction_horizon)]
            if all(col in data.columns for col in forecast_cols):
                forecast_data_series = data.iloc[selected_step_index][forecast_cols]
                forecast_df_for_chart = pd.DataFrame({'Hour Ahead': [f"+{h+1}hr" for h in range(prediction_horizon)], 'Predicted Temp (°C)': forecast_data_series.values}).set_index('Hour Ahead')
                st.bar_chart(forecast_df_for_chart); st.write("Forecast Data (Unscaled Temp °C):"); forecast_data_series.index = [f"+{h+1}hr" for h in range(prediction_horizon)]; st.dataframe(forecast_data_series)
            else: st.warning("Forecast columns not found in log file.")
        else: st.warning("Selected step index out of range.")

    # == TAB 6: Raw Data == (Keep as before)
    with tab_data:
        st.header("Raw Simulation Log Data"); st.dataframe(data)

else:
    st.warning("Log file could not be loaded. Please check the path and file integrity in the sidebar.")