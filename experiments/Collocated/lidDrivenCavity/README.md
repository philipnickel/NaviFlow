# Lid-Driven Cavity Flow at Re=100

This experiment demonstrates the simulation of lid-driven cavity flow at Reynolds number 100 using the SIMPLE algorithm implemented in NaviFlow.

## Problem Description

The lid-driven cavity is a classic CFD validation case:
- Square domain (1x1)
- No-slip conditions on all walls
- Top wall (lid) moves at constant velocity (u=1, v=0)
- Reynolds number = 100 (density = 1.0, viscosity = 0.01, lid velocity = 1.0, domain size = 1.0)

## Files

- `simulation.yaml`: Configuration file for the simulation
- `results/`: Directory containing output files
  - `u_field.npy`, `v_field.npy`, `p_field.npy`: NumPy arrays containing field values
  - `metadata.yaml`: Statistics and metadata for the simulation
  - `plots/`: Visualizations of the fields

## Running the Simulation

### Using the main entry point (as per usage.md)

The recommended way to run the simulation is using the main.py entry point:

```bash
# Run a single iteration
python main.py -config experiments/lid_driven_cavity_re100/simulation.yaml

# Run multiple iterations (e.g., 10)
python main.py -config experiments/lid_driven_cavity_re100/simulation.yaml -iterations 10
```

### Alternative methods

You can also use the dedicated scripts for a single iteration or visualization:

```bash
# Run a single iteration
python run_single_iteration.py

# Visualize the results
python visualize_results.py
```

## Initial Results (Single Iteration)

After just one iteration, the flow field already shows some expected features:
- Maximum u-velocity: 2.59 (will decrease with more iterations)
- Minimum v-velocity: -2.07 (downward flow near the right wall)
- Developing circulation pattern visible in vector field

### Key Statistics

```
u min/max: -0.6024/2.5913
v min/max: -2.0700/0.1828
p min/max: 0.0095/0.0429
Velocity magnitude min/max: 0.0000/3.3166
```

## Next Steps

For full convergence:
1. Run multiple iterations (1000+) using: `python main.py -config experiments/lid_driven_cavity_re100/simulation.yaml -iterations 1000`
2. Monitor residuals for convergence
3. Compare against Ghia et al. benchmark data
4. Repeat with finer meshes to assess grid independence 