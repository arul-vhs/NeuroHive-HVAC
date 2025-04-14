# multi_zone_sim.py
import numpy as np

# --- Shared Simulation Constants ---
HEAT_CAPACITY_AIR = 1005  # J/(kg*K)
AIR_DENSITY = 1.225       # kg/m^3
TIME_STEP = 3600          # seconds (1 hour) - Keep consistent for now

# --- Zone Definitions ---
# Define parameters for each zone
# We'll use a list of dictionaries
# Max cooling is the max power (Watts) that can be directed to this zone
zone_params = [
    {
        "name": "Server Room",
        "volume": 3 * 3 * 3,      # m^3 (Smaller)
        "wall_u_value": 1.0,    # W/(m^2*K) (Better insulation)
        "internal_gain": 1000,  # W (High constant gain)
        "criticality": 10.0,    # High criticality score
        "max_cooling_power": 4000 # W (Dedicated unit?)
    },
    {
        "name": "Office",
        "volume": 6 * 5 * 3,      # m^3 (Larger)
        "wall_u_value": 2.0,    # W/(m^2*K) (Average insulation)
        "internal_gain": 200,   # W (Moderate base gain, higher if occupied)
        "criticality": 5.0,     # Medium criticality
        "max_cooling_power": 5000 # W
    },
    {
        "name": "Hallway",
        "volume": 10 * 2 * 3,     # m^3 (Long)
        "wall_u_value": 2.5,    # W/(m^2*K) (Poorer insulation)
        "internal_gain": 50,    # W (Low gain)
        "criticality": 1.0,     # Low criticality
        "max_cooling_power": 3000 # W
    }
]

# Calculate derived parameters once
for zone in zone_params:
    # Simplified wall area calculation (adjust if needed)
    # Assuming approx square/rectangular for simplicity
    l = w = np.cbrt(zone["volume"] / 3) # Estimate length/width from volume/height
    h = 3
    zone["wall_area"] = 2*(l*h) + 2*(w*h) + l*w # 4 walls + ceiling
    zone["thermal_resistance"] = 1 / (zone["wall_u_value"] * zone["wall_area"]) # K/W
    zone["thermal_mass"] = zone["volume"] * AIR_DENSITY * HEAT_CAPACITY_AIR * 10 # Added multiplier for stability

num_zones = len(zone_params)
print(f"Defined {num_zones} zones.")

# --- Multi-Zone Simulation Step ---
def simulate_multi_zone_step(current_temps, outdoor_temp, cooling_allocations):
    """
    Simulates one time step for multiple zones.

    Args:
        current_temps (np.array): Array of current temperatures for each zone [C].
        outdoor_temp (float): Current outdoor temperature [C].
        cooling_allocations (np.array): Array of cooling power applied to each zone [W].
                                       Should be positive values.

    Returns:
        np.array: Array of new temperatures for each zone after TIME_STEP [C].
    """
    new_temps = np.zeros(num_zones)
    delta_temps = np.zeros(num_zones) # For debugging/info

    for i in range(num_zones):
        zone = zone_params[i]
        current_t = current_temps[i]
        allocated_cooling = cooling_allocations[i]

        # Ensure allocated cooling doesn't exceed max for the zone
        allocated_cooling = min(allocated_cooling, zone["max_cooling_power"])
        if allocated_cooling < 0: allocated_cooling = 0 # Cannot allocate negative cooling

        # Calculate heat flows
        heat_flow_external = (outdoor_temp - current_t) / zone["thermal_resistance"] # Watts
        internal_gain = zone["internal_gain"] # Simplified: use fixed gain for now

        # Net heat flow (External + Internal - Cooling)
        net_heat_flow = heat_flow_external + internal_gain - allocated_cooling

        # Calculate temperature change
        delta_t = (net_heat_flow * TIME_STEP) / zone["thermal_mass"]
        new_temps[i] = current_t + delta_t
        delta_temps[i] = delta_t

    return new_temps, delta_temps


# --- Example Usage (for testing this file) ---
if __name__ == '__main__':
    print("\nTesting Multi-Zone Simulation Step...")
    # Initial state
    current_temps = np.array([26.0, 24.0, 25.0]) # Server room hot, Office okay, Hallway warm
    outdoor_temp = 30.0 # Hot day
    # Example Allocation: Apply more cooling to server room, some to office, less to hallway
    # Total hypothetical cooling needed = 6000W
    alloc_proportions = np.array([0.5, 0.35, 0.15]) # Proportions decided by PSO
    total_cooling_power = 6000 # W - This value would come from the RL agent eventually
    cooling_allocations = alloc_proportions * total_cooling_power

    print(f"Initial Temps: {current_temps}")
    print(f"Outdoor Temp: {outdoor_temp}")
    print(f"Cooling Allocations (W): {cooling_allocations}")

    new_temps, delta_temps = simulate_multi_zone_step(current_temps, outdoor_temp, cooling_allocations)

    print(f"Delta Temps: {delta_temps}")
    print(f"New Temps: {new_temps}")

    # Test edge case: zero allocation
    cooling_allocations_zero = np.zeros(num_zones)
    print(f"\nTesting with zero cooling:")
    print(f"Cooling Allocations (W): {cooling_allocations_zero}")
    new_temps_zero, delta_temps_zero = simulate_multi_zone_step(current_temps, outdoor_temp, cooling_allocations_zero)
    print(f"Delta Temps: {delta_temps_zero}")
    print(f"New Temps: {new_temps_zero}") # Should see temps rise due to outdoor/internal gain