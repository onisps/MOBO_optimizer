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
from omegaconf import DictConfig
from utils.project_utils import (setup_logger, cleanup_logger, create_results_folder, save_optimization_summary)
from utils.global_variable import *
from utils.problem import init_procedure, LeafletProblem

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
        obj1 = -np.sum((X / 2) ** 2, axis=1)
        obj2 = -np.sum((X / 3 - 2) ** 2, axis=1)
        obj3 = -np.sum((X / 12 + 2) ** 1/3, axis=1)
        output = np.vstack([obj1, obj2, obj3]).T
        return output

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
            result = LeafletProblem.run_procedure(self=self.problem, params=parameters)
        elif problem_name == 'leaflet_contact':
            result = LeafletProblem.run_procedure(self=self.problem, params=parameters)

        out = np.array([[result['objectives'][name]] for name in self.obj_names]).T
        return out


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
            "chosen_actions": optimizer.chosen_actions,
            "fx": optimizer.fx,
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
        fx_array = np.array(optimizer.fx)
        current_min = fx_array.min(axis=0)
        self.min_values.append(current_min)

        if generation % self.interval_backup == 0:
            self.save_state(optimizer, generation)

        if generation % self.interval_picture == 0:
            self.plot_convergence(generation)


@hydra.main(config_path="configuration", config_name="config_leaf", version_base=None)
def main(cfg:DictConfig) -> None:

    parameters = {k: tuple(v) for k, v in cfg.parameters.items()}
    objectives = cfg.objectives

    print("\nConverted parameters (as tuples):")
    print(parameters)
    print("\nObjectives:")
    print(objectives)

    #parce config.yaml
    num_candidates = cfg.optimizer.num_candidates
    initial_samples = cfg.optimizer.initial_samples
    batch_size = cfg.optimizer.batch_size
    min_improvement = cfg.optimizer.termination_parameters.min_improvement
    no_improvement_generations = cfg.optimizer.termination_parameters.no_improvement_generations

    set_cpus(cfg.Abaqus.abq_cpus)
    set_tangent_behavior(cfg.Abaqus.tangent_behavior)
    set_normal_behavior(cfg.Abaqus.normal_behavior)

    set_DIA(cfg.problem_definition.DIA)
    set_Lift(cfg.problem_definition.Lift)
    set_SEC(cfg.problem_definition.SEC)
    set_EM(cfg.problem_definition.EM)
    set_density(cfg.problem_definition.Dens)
    set_material_name(cfg.problem_definition.material_name)
    set_mesh_step(cfg.problem_definition.mesh_step)
    set_valve_position(cfg.problem_definition.position)
    set_problem_name(cfg.problem_definition.problem_name)
    set_base_name(cfg.problem_definition.problem_name)
    set_s_lim(cfg.problem_definition.s_lim)

    basic_stdout = sys.stdout
    basic_stderr = sys.stderr
    basic_folder_path = create_results_folder(base_folder='results')
    logger = setup_logger(basic_folder_path)
    blackbox = Problem(parameters, objectives)
    np.random.seed(0)
    x_unscaled = np.random.rand(num_candidates, len(parameters))

    # PhysBO optimizer setup
    optimizer = physbo.search.discrete_multi.policy(test_X=x_unscaled, num_objectives=len(objectives))
    optimizer.set_seed(0)

    # Setup callback
    callback = PhysBOCallback(objectives=objectives, folder_path=basic_folder_path)
    # simulator = Simulator(blackbox)

    start_time = datetime.datetime.now()

    def simulator(indices):
        x = x_unscaled[indices]
        x_scaled = blackbox.evaluate(x)
        return x_scaled
    optimizer.random_search(max_num_probes=initial_samples, simulator=simulator, is_disp=False)

    # Bayesian search loop with robust termination criteria
    pareto_front_history = []
    generation = 0
    best_objective = np.inf
    print("[BayesOpt] Starting MOBO iterations...")
    while True:
        generation += 1
        res = optimizer.bayes_search(
            max_num_probes=initial_samples,
            simulator=simulator,
            num_search_each_probe=batch_size,
            is_disp=False,
            score='HVPI', # acquisition-функция для MOBO. Hypervolume-based Probability of Improvement
            interval=0,  # переобучение гиперпараметров только в начале
            num_rand_basis=0 # обычный GP (точнее, но медленнее)
        )

        # Check termination criteria
        current_best = np.min(res.fx)
        improvement = best_objective - current_best
        callback.notify(res, generation)
        pareto_y = res.pareto.front
        pareto_front_history.append(pareto_y)

        # Check improvement criterion
        if improvement > min_improvement:
            best_objective = current_best
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f'No improvement over {no_improve_count} generations.')

        # Check no improvement criterion
        if no_improve_count >= no_improvement_generations:
            print(f"Termination: no improvement over {no_improvement_generations} generations.")
            break
        print(f"[Iter {generation}] Pareto front size: {len(pareto_y)}")

    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    final_pareto = pareto_front_history[-1]

    # Collect optimization results
    chosen_indices = res.chosen_actions
    chosen_x_scaled = x_unscaled[chosen_indices]
    chosen_x = blackbox.scale_parameters(chosen_x_scaled)
    objectives_data = np.array(res.fx)

    # --- Create generation labels ---
    generations = (
            [0] * initial_samples +
            sum([[i] * batch_size for i in range(1, generation)], [])
    )
    generations = generations[:len(res.chosen_actions)]

    # Correct implementation to build DataFrame:
    history_df = pd.DataFrame(
        chosen_x[:len(generations)],
        columns=parameters.keys()
    )

    # Correctly indexing objective values:
    for i, obj in enumerate(objectives):
        history_df[obj] = objectives_data[:len(generations), i]

    history_df["generation"] = generations


    X_df = pd.DataFrame(chosen_x, columns=parameters.keys())
    X_df.to_excel(os.path.join(basic_folder_path,'x.xlsx'))
    X_df.to_csv(os.path.join(basic_folder_path,'x.csv'))
    F_df = pd.DataFrame(objectives_data, columns=objectives)
    F_df.to_excel(os.path.join(basic_folder_path,'F.xlsx'))
    F_df.to_csv(os.path.join(basic_folder_path,'F.csv'))
    G_df = pd.DataFrame(np.zeros((len(X_df), 0)))  # no constraints here, empty DataFrame

    # Save CSV for visualization
    history_csv_path = os.path.join(basic_folder_path, 'history.csv')
    history_excel_path = os.path.join(basic_folder_path, 'history.xlsx')
    history_df.to_csv(history_csv_path, index=False)
    history_df.to_excel(history_excel_path, index=False)

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
        termination_params={'max_iter': generation},
        detailed_algo_params={'method': 'PhysBO', 'criterion': 'EI'}
    )

    # --------------------------
    # Visualization plots
    # --------------------------
    from utils.visualize import (
        plot_objective_minimization,
        plot_objective_convergence,
        plot_objectives_vs_parameters,
        plot_parallel_coordinates,
        plot_best_objectives
    )

    plot_objective_minimization(history_df, basic_folder_path)
    plot_objective_convergence(history_df, objectives, basic_folder_path)
    plot_objectives_vs_parameters(history_df,parameters.keys(), objectives, basic_folder_path)
    plot_parallel_coordinates(history_df[parameters.keys()], G_df, history_df[objectives], objectives, basic_folder_path)
    plot_best_objectives(history_df[objectives], pd.DataFrame(final_pareto, columns=objectives), basic_folder_path)
    print(f"Optimization and visualization completed successfully.\nResults saved to: {basic_folder_path}")

    cleanup_logger(logger)
    del logger
    sys.stdout = basic_stdout
    sys.stderr = basic_stderr

if __name__ == '__main__':
    main()


