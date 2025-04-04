import os
from typing import Tuple, Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import matplotlib.colors as mcolors
import matplotlib as mpl
from pymoo.visualization.scatter import Scatter
from pymoo.decomposition.asf import ASF



def plot_pareto_front_comparison(
        analytical_data: np.ndarray,
        optimization_data: pd.DataFrame,
        objectives: list[str],
        folder_path: str,
        filename: str = 'pareto_front_comparison.png'
):
    """
    Plots the Pareto front comparing analytical (theoretical) data with optimization results.

    Args:
        analytical_data (np.ndarray): Analytical Pareto front data (array with shape [n_samples, 2]).
        optimization_data (pd.DataFrame): Optimization results DataFrame containing objective values.
        objectives (list[str]): List containing names of two objectives.
        folder_path (str): Directory to save the plot.
        filename (str): Name of the file for the saved plot.

    Returns:
        None: Saves the Pareto front comparison plot to the specified folder.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(
        optimization_data[objectives[0]], 
        optimization_data[objectives[1]],
        label='Optimization Data',
        color='blue',
        alpha=0.6,
        edgecolors='k'
    )
    plt.plot(
        analytical_data[:, 0], 
        analytical_data[:, 1],
        label='Analytical Pareto Front',
        color='red',
        linewidth=2
    )

    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.title('Comparison of Analytical and Optimization Pareto Fronts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(folder_path, filename)
    plt.savefig(save_path)
    plt.close()

def plot_objective_minimization(
        history_df: pd.DataFrame,
        folder_path: str
) -> None:
    """
    Creates a plot showing the minimum objective values over generations and saves it to a specified folder.

    Args:
        history_df (pd.DataFrame): The DataFrame containing optimization history.
        folder_path (str): The directory path to save the plot.

    Returns:
        None: This function creates a Seaborn plot and saves it to the specified file.
    """
    objective_columns = history_df.filter(like='objective').columns
    objectives_min_per_generation = history_df.groupby('generation')[objective_columns].min()
    objectives_over_time = objectives_min_per_generation.min(axis=1).tolist()
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 5))
    sns.lineplot(
        x=range(1, len(objectives_over_time) + 1),
        y=objectives_over_time,
        marker='.',
        linestyle='-',
        color='b',
        markersize=10
    )
    plt.title("Objective Minimization Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'convergence_by_objectives.png'))
    plt.close()

def plot_objective_convergence(
        history_df: pd.DataFrame,
        objectives: list[str],
        folder_path: str
) -> None:
    """
    Plots convergence of objectives over generations, with optional modes to visualize best objective values.

    Args:
        history_df (pd.DataFrame): DataFrame containing optimization history.
        folder_path (str): Directory path to save the plot.

    Returns:
        None: This function creates and saves a plot showing objective convergence over generations.
    """
    unique_generations = sorted(history_df['generation'].unique())
    objective_columns = history_df[objectives].columns
    num_objectives = len(objective_columns)
    fig, axes = plt.subplots(1, num_objectives, figsize=(15, 5), sharey=False)
    if num_objectives == 1:
        axes = [axes]
    for idx, obj_col in enumerate(objective_columns):
        if obj_col.lower() == 'lmn_open':
            obj_col_print = '1 - LMN_open'
        else:
            obj_col_print = obj_col
        min_per_generation = history_df.groupby('generation')[obj_col].min()
        sns.lineplot(x=unique_generations, y=min_per_generation, ax=axes[idx],
                     marker='o', color='b', markeredgecolor=None, label=f"Best {obj_col_print}")
        axes[idx].set_title(f"Convergence of ({obj_col_print})")
        axes[idx].set_xlabel("Generation")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objective_convergence.png'))
    plt.close()


def plot_objectives_vs_parameters(
        history_df: pd.DataFrame,
        parameters: [str],
        objectives: [str],
        folder_path: str
) -> None:
    """
    Plots scatter plots of each objective against each parameter in subplots.

    Args:
        history_df (pd.DataFrame): DataFrame history of optimisation.
        parameters (list string): List of parameters names.
        objectives (list string): List of objectives names.
        folder_path (str): Directory path to save the plot.

    Returns:
        None: This function creates scatter plots and saves them to the specified file.
    """
    X = history_df[parameters]
    F = history_df[objectives]
    num_params = len(X.columns)
    num_objectives = len(F.columns)

    fig, axes = plt.subplots(num_params, num_objectives, figsize=(15, 10), sharex=False, sharey=False)

    if num_params == 1 or num_objectives == 1:
        axes = np.array(axes).reshape((num_params, num_objectives))

    for param_idx, param in enumerate(X.columns):
        for obj_idx, obj in enumerate(F.columns):
            ax = axes[param_idx, obj_idx]
            sns.scatterplot(
                x=X[param],
                y=F[obj],
                ax=ax,
                s=30,
                color='b',
                alpha=0.8,
                hue=False,
                legend=False
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'objectives_vs_parameters.png'))
    plt.close()


def plot_constrains_vs_parameters(
        X: pd.DataFrame,
        G: pd.DataFrame,
        folder_path: str
) -> None:
    """
    Plots scatter plots of each constrain against each parameter in subplots.

    Args:
        X (pd.DataFrame): DataFrame containing parameter values.
        G (pd.DataFrame): DataFrame containing constrain values.
        folder_path (str): Directory path to save the plot.

    Returns:
        None: This function creates scatter plots and saves them to the specified file.
    """

    num_params = len(X.columns)
    num_constrains = len(G.columns)

    fig, axes = plt.subplots(num_params, num_constrains, figsize=(20, 10), sharex=False, sharey=False)

    if num_params == 1 or num_constrains == 1:
        axes = np.array(axes).reshape((num_params, num_constrains))

    for param_idx, param in enumerate(X.columns):
        for obj_idx, constr in enumerate(G.columns):
            ax = axes[param_idx, obj_idx]
            sns.scatterplot(
                x=X[param],
                y=G[constr],
                ax=ax,
                s=30,
                color='b',
                alpha=0.8,
                hue=False,
                legend=False
            )

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'constrain_vs_parameters.png'))
    plt.close()


def plot_parallel_coordinates(
        X: pd.DataFrame,
        G: pd.DataFrame,
        F: pd.DataFrame,
        objectives: list[str],
        folder_path: str,
        file_name: str = 'parallel_coordinates.html'
) -> None:
    """
    Creates an interactive parallel coordinates plot for parameters, constraints,
    and objectives, and saves it to a specified folder.

    Args:
        X (pd.DataFrame): DataFrame containing parameter values.
        G (pd.DataFrame): DataFrame containing constraint values.
        F (pd.DataFrame): DataFrame containing objective values.
        folder_path (str): Directory path to save the plot.
        file_name (str, optional): Name of the file for saving the plot. Defaults to 'parallel_coordinates.html'.

    Returns:
        None: This function creates and saves an interactive parallel coordinates plot.
    """

    combined_data = pd.concat([X, G, F], axis=1)
    score = combined_data[objectives[0]]
    for obj in objectives[1:-1]:
        score *= combined_data[obj]

    fig = px.parallel_coordinates(
        combined_data,
        dimensions=combined_data.columns,
        color=score,
        color_continuous_scale=px.colors.diverging.Tealrose,
        labels={col: col for col in combined_data.columns},
        title="Parallel Coordinates Plot",
        width=1024,
        height=768
    )
    plot_path = os.path.join(folder_path, file_name)
    fig.write_html(plot_path)



def plot_best_objectives(
        F: pd.DataFrame,
        pareto_front: pd.DataFrame,
        folder_path: str,
        weights: list[float] or str = 'equal'
) -> None:
    """
    Finds the best trade-off in multi-objective maximization based on a weighted Chebyshev norm
    and creates subplots for each unique pair of objectives to visualize the best solution.

    Args:
        F (pd.DataFrame): DataFrame with objective values (to be maximized).
        pareto_front (pd.DataFrame): DataFrame with pareto front.
        folder_path (str): Directory to save plots.
        weights (list[float] or str): Objective weights or 'equal'.

    Returns:
        None
    """
    num_objectives = F.shape[1]

    if weights == 'equal':
        weights = [1 / num_objectives] * num_objectives
    elif isinstance(weights, list) and len(weights) == num_objectives:
        if not np.isclose(sum(weights), 1):
            raise ValueError("List of weights must sum to 1.")
    else:
        raise ValueError("Weights must be either 'equal' or a list of correct length.")

    weights = np.array(weights)

    # Нормализуем цели от 0 до 1 (по каждому столбцу)
    ideal = F.min(axis=0)  # минимум по каждому столбцу
    nadir = F.max(axis=0)  # максимум по каждому столбцу
    normalized_F = (F - ideal) / (nadir - ideal + 1e-8)

    # Вычисляем "Chebyshev-like" метрику (взвешенное максимальное значение)
    scores = (normalized_F / weights).max(axis=1)

    best_index = scores.idxmax()  # Минимизируем по максимуму — наилучший компромисс

    # Визуализация
    subplot_count = (num_objectives * (num_objectives - 1)) // 2
    fig, axes = plt.subplots(1, subplot_count, figsize=(5 * subplot_count, 5))
    plot_idx = 0

    for i in range(num_objectives):
        for j in range(i + 1, num_objectives):
            ax = axes[plot_idx] if subplot_count > 1 else axes
            sns.scatterplot(
                data=F,
                x=F.columns[i],
                y=F.columns[j],
                label="All points",
                s=30,
                ax=ax,
                color='blue',
                alpha=0.6
            )
            sns.scatterplot(
                data=F.loc[[best_index]],
                x=F.columns[i],
                y=F.columns[j],
                label="Best trade-off",
                s=200,
                marker="x",
                color="red",
                ax=ax
            )
            sns.scatterplot(
                data=pareto_front,
                x=pareto_front.columns[i],
                y=pareto_front.columns[j],
                label="Pareto front",
                s=30,
                ax=ax,
                color='orangered',
                alpha=0.6
            )
            ax.set_title(f"{F.columns[i]} vs {F.columns[j]}")
            ax.set_xlabel(F.columns[i])
            ax.set_ylabel(F.columns[j])
            plot_idx += 1

    plt.tight_layout()
    os.makedirs(folder_path, exist_ok=True)
    plt.savefig(os.path.join(folder_path, 'objective_space_subplots.png'))
    plt.close()


def load_optimization_results(
        folder_path: str,
        csv_files: Dict[str, str]
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Load optimization results from CSV files in a specified folder.

    Args:
        folder_path (str): The path to the folder where CSV files are stored.
        csv_files (Dict[str, str]): A dictionary mapping DataFrame names to CSV file names.

    Returns:
        Dict[str, Optional[pd.DataFrame]]: A dictionary with keys representing the names of DataFrames
            and values being the loaded DataFrames or `None` if the file is not found.
    """
    dataframes = {}
    for df_name, csv_file in csv_files.items():
        try:
            dataframes[df_name] = pd.read_csv(os.path.join(folder_path, csv_file), index_col=0)
        except FileNotFoundError:
            print(f"Warning: '{csv_file}' not found in '{folder_path}'.")
            dataframes[df_name] = None

    return dataframes
