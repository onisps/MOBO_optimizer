import sys
import os
import numpy as np
import pandas as pd
import physbo
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.visualize import (
    plot_objective_minimization,
    plot_objective_convergence,
    plot_objectives_vs_parameters,
    plot_parallel_coordinates,
    plot_best_objectives,
    plot_pareto_front_comparison
)
from utils.project_utils import (setup_logger, cleanup_logger, create_results_folder, save_optimization_summary)
from utils.global_variable import (set_problem_name, set_percent, set_cpus, set_base_name, set_s_lim, get_s_lim, set_id,
                                   set_dead_objects, set_mesh_step, set_valve_position, get_cpus, get_problem_name)
from utils.problem import init_procedure, Procedure

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



class Problem:
    problem = None

    def __init__(self, parameters, objectives):
        self.param_bounds = parameters
        self.parameters = parameters.items()
        self.param_names = parameters.keys()
        self.obj_names = objectives

    def scale_parameters(self, X_scaled):
        X_original = np.zeros_like(X_scaled)
        for idx, key in enumerate(self.param_bounds.keys()):
            min_val, max_val = self.param_bounds[key]
            X_original[:, idx] = X_scaled[:, idx] * (max_val - min_val) + min_val
        return X_original

    def evaluate(self, x):
        x_init = x
        x = self.scale_parameters(x)[0]
        self.problem = init_procedure(np.array(x))
        problem_name = get_problem_name().lower()
        cpus = get_cpus()
        parameters = np.array(x)
        if problem_name == 'leaflet_single':
            result = Procedure.run_procedure(self=self.problem, params=parameters)
            objective_values = result['objectives']
            objectives_dict = {
                "1 - LMN_open": objective_values['1 - LMN_open'],
                "LMN_open": objective_values['LMN_open'],
                "LMN_closed": objective_values['LMN_closed'],
                "Smax": objective_values['Smax']
            }
            # print(f'obj: {objectives_dict}')
            constraint_values = result['constraints']
            constraints_dict = {
                "VMS-Smax": constraint_values['VMS-Smax']
            }
            # print(f'cons: {constraints_dict}')
        elif problem_name == 'leaflet_contact':
            # result = Procedure.run_procedure(self=self.problem, params=parameters)
            result =    {'objectives': {'1 - LMN_open': np.random.rand(),
               'LMN_open':  np.random.rand(),
               'LMN_closed': np.random.rand(),
               'Smax - Slim': get_s_lim() - np.random.rand()*5,
               'HELI': np.random.rand()},
              'constraints': {'VMS-Smax': get_s_lim() - 3*np.random.rand()}}
        return np.array([[result['objectives'][name]] for name in self.obj_names]).T


@hydra.main(config_path="configuration", config_name="config", version_base=None)
def main(cfg:DictConfig) -> None:
    # print("loaded with hydra:")
    # print(OmegaConf.to_yaml(cfg))
    print(cfg)

    parameters = {k: tuple(v) for k, v in cfg.parameters.items()}
    objectives = cfg.objectives

    print("\nConverted parameters (as tuples):")
    print(parameters)
    print("\nObjectives:")
    print(objectives)

    num_candidates = cfg.num_candidates
    initial_samples = cfg.initial_samples
    n_iter = cfg.n_iter

    basic_stdout = sys.stdout
    basic_stderr = sys.stderr
    basic_folder_path = create_results_folder(base_folder='results')
    print(f"folder path > {basic_folder_path}")

    # logging
    logger = setup_logger(basic_folder_path)


    blackbox = Problem(parameters, objectives)
    # blackbox = BlackBoxFunction(parameters, objectives)

    X_scaled = np.random.rand(num_candidates, len(parameters))

    # --------------------------
    # PhysBO optimizer setup
    # --------------------------
    optimizer = physbo.search.discrete.policy(test_X=X_scaled)

    def simulator(indices):
        X = X_scaled[indices]
        return blackbox.evaluate(X)[:, 0]

    np.random.seed(0)
    optimizer.random_search(max_num_probes=initial_samples, simulator=simulator)

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
    for objective, objective_data in zip(objectives, objectives_data.T):
        history_df[objective] = objective_data
    history_df['generation'] = np.arange(len(history_df))

    # results_dir = 'optimization_results'
    os.makedirs(basic_folder_path, exist_ok=True)

    X_df = pd.DataFrame(chosen_X, columns=parameters.keys())
    F_df = pd.DataFrame(objectives_data, columns=objectives)
    G_df = pd.DataFrame(np.zeros((len(X_df), 0)))  # no constraints here, empty DataFrame

    # Save CSV for visualization
    history_csv_path = os.path.join(basic_folder_path, 'history.csv')
    history_df.to_csv(history_csv_path, index=False)

    # Call the saving function
    best_index = optimizer.history.fx.argmin()
    save_optimization_summary(
        type_of_run='PhysBO contact',
        folder_path=basic_folder_path,
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

    # Analytical Pareto front calculation
    def generate_analytical_pareto_front(num_points=50):
        lambdas = np.linspace(0, 1, num_points)
        f1 = 20 * (1 - lambdas) ** 2
        f2 = 20 * lambdas ** 2
        analytical_pareto = np.column_stack([f1, f2])
        return analytical_pareto

    plot_objective_minimization(history_df, basic_folder_path)

    plot_objective_convergence(history_df, objectives, basic_folder_path)

    plot_objectives_vs_parameters(X_df, F_df, basic_folder_path)

    plot_parallel_coordinates(X_df, G_df, F_df, objectives, basic_folder_path)

    plot_best_objectives(F_df, basic_folder_path)

    analytical_pareto_data = generate_analytical_pareto_front()
    plot_pareto_front_comparison(
        analytical_data=analytical_pareto_data,
        optimization_data=F_df,
        objectives=objectives,
        folder_path=basic_folder_path
    )

    print(f"Optimization and visualization completed successfully.\nResults saved to: {basic_folder_path}")

    cleanup_logger(logger)
    del logger
    sys.stdout = basic_stdout
    sys.stderr = basic_stderr

if __name__ == '__main__':
    set_mesh_step(0.4)
    set_valve_position('mitr') # can be 'mitr'
    problem_name = 'leaflet_contact'
    set_problem_name(problem_name)
    set_base_name('Mitral_test')
    set_s_lim(3.23)  # Formlabs elastic 50A
    set_cpus(3)  # 3 cpu cores shows better results then 8 cores. 260sec vs 531sec


    main()


