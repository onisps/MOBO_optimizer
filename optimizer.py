import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import physbo
import pickle
import datetime
import hydra
from matplotlib import pyplot as plt
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
        x = self.scale_parameters(X_scaled)
        n = x.shape[1]
        obj1 = 1 - np.exp(-1 * np.sum((x - 1/np.sqrt(n)) ** 2, axis = 1))
        obj2 = 1 - np.exp(-1 * np.sum((x + 1/np.sqrt(n)) ** 2, axis = 1))
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



class PhysBOCallback:
    def __init__(self, objectives, folder_path='./utils/logs/', interval_backup=10, interval_picture=2):
        self.objectives = objectives
        self.folder_path = folder_path
        os.makedirs(self.folder_path, exist_ok=True)
        self.interval_backup = interval_backup
        self.interval_picture = interval_picture
        self.min_values = []

    def save_state(self, optimizer, generation):
        filepath = os.path.join(self.folder_path, f'checkpoint_gen_{generation:03d}.pkl')
        state = {
            "chosen_actions": optimizer.history.chosen_actions,
            "fx": optimizer.history.fx,
            "generation": generation
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"[Callback] Saved state at generation {generation}")

    def plot_convergence(self, generation):
        fx_array = np.array(self.min_values)  # shape: (generations, n_objectives)
        gens = np.arange(fx_array.shape[0])

        sns.set(style="whitegrid")
        num_objectives = fx_array.shape[1]
        fig, axes = plt.subplots(1, num_objectives, figsize=(15, 5))

        if num_objectives == 1:
            axes = [axes]

        for i, obj_name in enumerate(self.objectives):
            y_values = fx_array[:, i]
            sns.lineplot(x=gens, y=y_values, ax=axes[i],
                         marker='o', color='b', label=f"Best {obj_name}")
            if len(y_values) > 2:
                axes[i].set_title(f"Convergence of {obj_name} | Δ={abs(y_values[-2] - y_values[-1]):.2e}")
            else:
                axes[i].set_title(f"Convergence of {obj_name}")
            axes[i].set_xlabel("Generation")
            axes[i].set_ylabel(obj_name)

        fig.tight_layout()
        plt.savefig(os.path.join(self.folder_path, 'intime_convergence.png'))
        plt.close()
        print(f"[Callback] Saved convergence plot at generation {generation}")

    def notify(self, optimizer, generation):
        # Compute min for each objective
        fx_array = np.array(optimizer.history.fx)
        current_min = fx_array.min(axis=0)
        self.min_values.append(current_min)

        if generation % self.interval_backup == 0:
            self.save_state(optimizer, generation)

        if generation % self.interval_picture == 0:
            self.plot_convergence(generation)

@hydra.main(config_path="configuration", config_name="config_test", version_base=None)
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


    # blackbox = Problem(parameters, objectives)
    blackbox = BlackBoxFunction(parameters, objectives)

    X_scaled = np.random.rand(num_candidates, len(parameters))

    # --------------------------
    # PhysBO optimizer setup
    # --------------------------

    optimizer = physbo.search.discrete_multi.policy(test_X=X_scaled, num_objectives=len(objectives))
    physbo.search.discrete_multi.policy.set_seed(seed=0)

    def simulator(indices):
        X = X_scaled[indices]
        return blackbox.evaluate(X)

    np.random.seed(0)
    optimizer.random_search(max_num_probes=initial_samples, simulator=simulator)

    # Setup callback
    callback = PhysBOCallback(objectives=objectives, folder_path=basic_folder_path)

    start_time = datetime.datetime.now()
    for i in range(1, n_iter + 1):
        optimizer.bayes_search(max_num_probes=1, simulator=simulator)
        callback.notify(optimizer, i)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()

    # --------------------------
    # Collect optimization results
    # --------------------------
    chosen_X_scaled = X_scaled[optimizer.history.chosen_actions]
    chosen_X = blackbox.scale_parameters(chosen_X_scaled)
    objectives_data = np.array(optimizer.history.fx)

    history_df = pd.DataFrame(chosen_X, columns=parameters.keys())
    for i, objective in enumerate(objectives):
        history_df[objective] = objectives_data[:,i]
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


