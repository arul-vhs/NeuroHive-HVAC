# pso_controller.py (Enhanced Predictive Fitness)

import numpy as np
import random

# Import zone params AND simulation function
try:
    from multi_zone_sim import zone_params, num_zones, simulate_multi_zone_step
    # Import constants needed for fitness (assuming they are in multi_zone_sim or hvac_env)
    # If not, define them here or pass them in. For now, assume defined globally/imported.
    from hvac_env import COMFORT_LOW, COMFORT_HIGH, COOLING_POWER_LOW, COOLING_POWER_HIGH
except ImportError:
    print("Error importing from multi_zone_sim.py or hvac_env.py.")
    # Dummy values
    zone_params = [{"criticality": 1, "name": f"Zone{i}", "max_cooling_power": 5000} for i in range(3)]; num_zones = 3
    def simulate_multi_zone_step(ct, ot, ca): return ct - np.random.rand(len(ct)), np.random.rand(len(ct))
    COMFORT_LOW = 21.0; COMFORT_HIGH = 24.0; COOLING_POWER_LOW = 3000; COOLING_POWER_HIGH = 6000

# --- PSO Constants ---
N_PARTICLES = 30; MAX_ITER = 50; W = 0.5; C1 = 1.5; C2 = 1.5

# --- Fitness Function Weights (TUNABLE - Emphasize Comfort, Penalize Overcooling/Energy) ---
W_COMFORT_HOT = 1.5   # Penalty weight for being too hot
W_COMFORT_COLD = 2.0  # STRONGER Penalty weight for being too cold (overcooling)
W_STRESS = 0.005      # Keep stress penalty very low
W_ENERGY_WASTE = 0.1  # Penalty for using much more power than needed? (Optional/Complex)

# --- Particle Class --- (Keep as before)
class Particle:
    def __init__(self): self.position = np.random.rand(num_zones); self.position = self.position / np.sum(self.position) if np.sum(self.position) > 0 else np.ones(num_zones)/num_zones; self.velocity = np.random.uniform(-0.1, 0.1, num_zones); self.pbest_position = self.position.copy(); self.pbest_value = float('inf'); self.current_fitness = float('inf')
    def update_velocity(self, gbest_position): r1=random.random(); r2=random.random(); cognitive=C1*r1*(self.pbest_position-self.position); social=C2*r2*(gbest_position-self.position); self.velocity = W*self.velocity + cognitive + social
    def update_position(self): self.position = self.position + self.velocity; self.position[self.position < 0] = 0.01

# --- Fitness Function (Enhanced Predictive Version) ---
def calculate_fitness(allocation_priorities, current_temps, outdoor_temp, total_cooling_power, comfort_low, comfort_high):
    """
    Calculates fitness based on predicted next state, penalizing overcooling more.
    """
    if np.sum(allocation_priorities) <= 1e-6: return float('inf')

    alloc_proportions = allocation_priorities / np.sum(allocation_priorities)
    zone_cooling_watts = alloc_proportions * total_cooling_power

    # Simulate the next state
    next_temps, _ = simulate_multi_zone_step(current_temps, outdoor_temp, zone_cooling_watts)

    # Calculate cost based on the predicted next state
    comfort_cost = 0
    for i in range(num_zones):
        next_t = next_temps[i]
        zone_criticality = zone_params[i]["criticality"]
        zone_penalty = 0
        # Penalize deviation outside band, with higher penalty for overcooling
        if next_t < comfort_low:
            zone_penalty = W_COMFORT_COLD * (comfort_low - next_t)**2 # Higher penalty below low
        elif next_t > comfort_high:
            zone_penalty = W_COMFORT_HOT * (next_t - comfort_high)**2 # Standard penalty above high
        comfort_cost += zone_criticality * zone_penalty

    # Stress cost (variance)
    stress_cost = np.var(alloc_proportions)

    # --- Optional Energy Waste Term ---
    # Estimate 'ideal' cooling needed to bring avg temp to midpoint? Very rough.
    # avg_current_temp = np.mean(current_temps)
    # target_mid = (comfort_low + comfort_high) / 2.0
    # estimated_delta_needed = target_mid - avg_current_temp # Negative if cooling needed
    # # This needs calibration based on thermal mass, timestep etc. - complex. Skip for now.
    # energy_waste_cost = 0
    # # ... logic to compare total_cooling_power to some estimate of actual need ...
    # --- End Optional ---

    # Total fitness - Sum of weighted costs
    fitness = comfort_cost + W_STRESS * stress_cost # Removed W_COMFORT multiplier, implicitly handled by W_COMFORT_HOT/COLD
    if not np.isfinite(fitness): fitness = float('inf')
    return fitness

# --- Swarm Class --- (Keep optimize/run_iteration calls consistent with new calculate_fitness args)
class Swarm:
    def __init__(self, n_particles=N_PARTICLES): self.particles=[Particle() for _ in range(n_particles)]; self.gbest_position=np.random.rand(num_zones); self.gbest_value=float('inf')
    def update_gbest(self): # ... (same as before) ...
        for p in self.particles:
            if p.pbest_value < self.gbest_value: self.gbest_value=p.pbest_value; self.gbest_position=p.pbest_position.copy()
    def run_iteration(self, current_temps, outdoor_temp, total_cooling_power, comfort_low, comfort_high): # Pass all args
        for p in self.particles:
            p.current_fitness = calculate_fitness(p.position, current_temps, outdoor_temp, total_cooling_power, comfort_low, comfort_high) # Pass all args
            if p.current_fitness < p.pbest_value: p.pbest_value = p.current_fitness; p.pbest_position = p.position.copy()
        self.update_gbest()
        for p in self.particles: p.update_velocity(self.gbest_position); p.update_position()
    def optimize(self, current_temps, outdoor_temp, total_cooling_power, comfort_low, comfort_high, max_iter=MAX_ITER): # Pass all args
        print(f"Running PSO for {max_iter} iterations...")
        for p in self.particles: # Initial fitness
             p.current_fitness = calculate_fitness(p.position, current_temps, outdoor_temp, total_cooling_power, comfort_low, comfort_high)
             if p.current_fitness < p.pbest_value: p.pbest_value = p.current_fitness; p.pbest_position = p.position.copy()
        self.update_gbest(); print(f"  Iter 0, Initial Best Fitness: {self.gbest_value:.4f}")
        for i in range(1, max_iter): # Main loop
            self.run_iteration(current_temps, outdoor_temp, total_cooling_power, comfort_low, comfort_high) # Pass all args
            if i % 10 == 0: print(f"  Iter {i}, Best Fitness: {self.gbest_value:.4f}")
        print(f"PSO finished. Final Best Fitness: {self.gbest_value:.4f}")
        if np.sum(self.gbest_position) > 1e-6: best_proportions = self.gbest_position / np.sum(self.gbest_position)
        else: best_proportions = np.ones(num_zones)/num_zones; print("Warning: gbest near zero.")
        print(f"Best Allocation Proportions: {best_proportions}")
        return best_proportions

# --- Example Usage --- (Keep as before)
if __name__ == '__main__':
    print("\nTesting PSO Controller (Enhanced Predictive Fitness)...")
    current_temps_test = np.array([26.0, 24.0, 25.0]); outdoor_temp_test = 30.0; total_cooling_needed_test = 6000
    comfort_l = 21.0; comfort_h = 24.0
    swarm = Swarm(n_particles=N_PARTICLES)
    best_allocation_proportions = swarm.optimize(current_temps_test, outdoor_temp_test, total_cooling_needed_test, comfort_l, comfort_h, max_iter=MAX_ITER)
    zone_cooling_watts = best_allocation_proportions * total_cooling_needed_test
    print(f"\nExample Usage:"); print(f"Total Cooling Need: {total_cooling_needed_test} W"); print(f"Calculated Zone Watts:")
    for i in range(num_zones): print(f"  {zone_params[i]['name']}: {zone_cooling_watts[i]:.2f} W (Prop: {best_allocation_proportions[i]:.3f})")
    sim_temps_next, _ = simulate_multi_zone_step(current_temps_test, outdoor_temp_test, zone_cooling_watts)
    print(f"Simulated Temps after 1 step w/ optimal allocation: {sim_temps_next}")