import numpy as np
import pandas as pd
import physbo
import os
import datetime
import openpyxl
from visualize import (
    plot_objective_minimization,
    plot_objective_convergence,
    plot_objectives_vs_parameters,
    plot_parallel_coordinates,
    plot_best_objectives
)

# --------------------------
# Black-box function class
# --------------------------
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

# --------------------------
# Optimization Parameters
# --------------------------
parameters = {
    'param1': (-5, 5),
    'param2': (0, 10),
    'param3': (1, 6),
    'param4': (-2, 3),
    'param5': (0, 1),
}

objectives = ['objective1', 'objective2']
blackbox = BlackBoxFunction(parameters, objectives)

num_candidates = 2000
X_scaled = np.random.rand(num_candidates, len(parameters))

# --------------------------
# PhysBO optimizer setup
# --------------------------
optimizer = physbo.search.discrete.policy(test_X=X_scaled)

def simulator(indices):
    X = X_scaled[indices]
    return blackbox.evaluate(X)[:, 0]

np.random.seed(0)
initial_samples = 15
optimizer.random_search(max_num_probes=initial_samples, simulator=simulator)

n_iter = 30
start_time = datetime.datetime.now()
for _ in range(n_iter):
    optimizer.bayes_search(max_num_probes=1, simulator=simulator, score='EI')
elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

# --------------------------
# Collect optimization results
# --------------------------
chosen_X_scaled = X_scaled[optimizer.history.chosen_actions]
chosen_X = blackbox.scale_parameters(chosen_X_scaled)
objectives_data = blackbox.evaluate(chosen_X_scaled)

history_df = pd.DataFrame(chosen_X, columns=parameters.keys())
history_df[objectives] = objectives_data
history_df['generation'] = np.arange(len(history_df))

results_dir = 'optimization_results'
os.makedirs(results_dir, exist_ok=True)

X_df = pd.DataFrame(chosen_X, columns=parameters.keys())
F_df = pd.DataFrame(objectives_data, columns=objectives)
G_df = pd.DataFrame(np.zeros((len(X_df), 0)))  # no constraints here, empty DataFrame

# Save CSV for visualization
history_csv_path = os.path.join(results_dir, 'history.csv')
history_df.to_csv(history_csv_path, index=False)

# --------------------------
# Saving Function (adapted)
# --------------------------
def save_optimization_summary(
    type_of_run: str = None,
    folder_path: str = None,
    best_index: int = None,
    elapsed_time: float = None,
    F: pd.DataFrame = None,
    X: pd.DataFrame = None,
    G: pd.DataFrame = None,
    history_df: pd.DataFrame = None,
    termination_params: dict = None,
    detailed_algo_params: dict = None
):
    base_folder = '/'.join(folder_path.split(os.path.sep)[:-1])
    summary_file = os.path.join(base_folder, 'optimization_summary.xlsx')

    if os.path.exists(summary_file):
        workbook = openpyxl.load_workbook(summary_file)
        worksheet = workbook.active
    else:
        workbook = openpyxl.Workbook()
        worksheet = workbook.active
        headers = (["Timestamp", "Type of run", "Generation", "Best Index", "Elapsed Time"] +
                   [f"F_{col}" for col in F.columns] +
                   [f"X_{col}" for col in X.columns] +
                   [f"G_{col}" for col in G.columns] +
                   [f"Term_{key}" for key in termination_params] +
                   list(detailed_algo_params.keys()) +
                   ['Folder_path'])
        worksheet.append(headers)

    generation_number = history_df['generation'].max()
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_F = F.iloc[best_index].tolist()
    best_X = X.iloc[best_index].tolist()
    best_G = G.iloc[best_index].tolist()

    termination_values = list(termination_params.values())
    detailed_algo_values = list(detailed_algo_params.values())

    new_row = [timestamp, type_of_run, generation_number, best_index, elapsed_time] + \
              best_F + best_X + best_G + termination_values + detailed_algo_values + [folder_path]

    worksheet.append(new_row)
    workbook.save(summary_file)

# Call the saving function
best_index = optimizer.history.fx.argmin()
save_optimization_summary(
    type_of_run='PhysBO',
    folder_path=results_dir,
    best_index=best_index,
    elapsed_time=elapsed_time,
    F=F_df,
    X=X_df,
    G=G_df,
    history_df=history_df,
    termination_params={'max_iter': n_iter},
    detailed_algo_params={'method': 'PhysBO', 'criterion': 'EI'}
)

# --------------------------
# Visualization plots
# --------------------------
plot_objective_minimization(history_df, results_dir)

plot_objective_convergence(history_df, objectives, results_dir)

plot_objectives_vs_parameters(X_df, F_df, results_dir)

plot_parallel_coordinates(X_df, G_df, F_df, objectives, results_dir)

plot_best_objectives(F_df, results_dir)

print(f"Optimization and visualization completed successfully.\nResults saved to: {results_dir}")
