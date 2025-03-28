import numpy as np
import pandas as pd
import physbo
import os
from utils.visualize import (
    create_pareto_front_plot,
    plot_objective_minimization,
    plot_pareto_with_trade_off
)

# Class defining the black-box function
class BlackBoxFunction:
    def __init__(self, param_bounds, objectives):
        self.param_bounds = param_bounds
        self.objectives = objectives
        
    def scale_parameters(self, X_scaled):
        X_original = np.zeros_like(X_scaled)
        for idx, key in enumerate(self.param_bounds.keys()):
            min_val, max_val = self.param_bounds[key]
            X_original[:, idx] = X_scaled[:, idx] * (max_val - min_val) + min_val
        return X_original
    
    def evaluate(self, X_scaled):
        X = self.scale_parameters(X_scaled)
        obj1 = np.sum(X**2, axis=1)
        obj2 = np.sum((X - 2)**2, axis=1)
        return np.vstack([obj1, obj2]).T

# Parameters and objectives
parameters = {
    'param1': (-5, 5),
    'param2': (0, 10),
    'param3': (1, 6),
    'param4': (-2, 3),
    'param5': (0, 1),
}

objectives = ['objective1', 'objective2']
blackbox = BlackBoxFunction(parameters, objectives)

# Search space
num_candidates = 2000
X_scaled = np.random.rand(num_candidates, len(parameters))

# Initialize PhysBO optimizer
optimizer = physbo.search.discrete.policy(test_X=X_scaled)

# Simulator using indices
def simulator(indices):
    X = X_scaled[indices]
    Y = blackbox.evaluate(X)
    return Y[:, 0]  # optimize by first objective

# Initial random sampling
np.random.seed(0)
initial_samples = 15
optimizer.random_search(max_num_probes=initial_samples, simulator=simulator)

# Bayesian optimization loop
n_iter = 30
for _ in range(n_iter):
    optimizer.bayes_search(max_num_probes=1, simulator=simulator, score='EI')

# Best parameters
best_sample_index = optimizer.history.fx.argmin()
best_x_scaled = X_scaled[best_sample_index]
best_x_original = blackbox.scale_parameters(best_x_scaled.reshape(1, -1))
best_objective_values = blackbox.evaluate(best_x_scaled.reshape(1, -1))

print("Optimal parameters:")
for idx, key in enumerate(parameters.keys()):
    print(f"{key}: {best_x_original[0, idx]}")

print("\nObjectives at optimum:")
for idx, obj in enumerate(objectives):
    print(f"{obj}: {best_objective_values[0, idx]}")

# -----------------------
# Prepare results for visualization
# -----------------------

# Creating a results directory
results_dir = 'optimization_results'
os.makedirs(results_dir, exist_ok=True)

# Create a DataFrame for history data (parameters and objectives)
history_df = pd.DataFrame(
    blackbox.scale_parameters(X_scaled[optimizer.history.chosen_actions]),
    columns=list(parameters.keys())
)

# Adding objectives
objectives_data = blackbox.evaluate(X_scaled[optimizer.history.chosen_actions])
history_df[objectives] = objectives_data

# Adding generation (iteration number)
history_df['generation'] = np.arange(len(history_df))

# Save history to CSV (for visualization functions)
history_csv_path = os.path.join(results_dir, 'history.csv')
history_df.to_csv(history_csv_path, index=False)

# -----------------------
# Visualization using provided module
# -----------------------

# Pareto front visualization
plot_objective_minimization(
    csv_path=history_csv_path,
    data=history_df[objectives],
    folder_path=results_dir,
    objectives=objectives,
    pymoo_problem='welded_beam'  # replace if needed
)

# Objective minimization (convergence)
plot_objective_minimization(
    history_df=history_df,
    folder_path=results_dir
)

# Pareto front with trade-off highlighted
plot_pareto_with_trade_off(
    history=history_df,
    F=history_df[objectives],
    objectives=objectives,
    folder_path=results_dir,
    weights='equal'  # or specify weights
)

print(f"\nVisualizations have been saved to: {results_dir}")
