import numpy as np
import hydra
from omegaconf import DictConfig
from physbo.search.discrete import policy


# Ваш класс BlackBoxFunction
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
        if X_scaled.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_scaled.shape}")

        X = self.scale_parameters(X_scaled)
        obj1 = np.sum((X / 2) ** 2, axis=1)
        obj2 = np.sum((X / 3 - 2) ** 2, axis=1)

        output = np.vstack([obj1, obj2]).T
        return output
# При необходимости импортировать PHYSBO или использовать собственную реализацию MOBO
# import physbo
# from physbo.search.discrete import policy

def latin_hypercube_sampling(n_samples, n_dims):
    return np.random.rand(n_samples, n_dims)

@hydra.main(config_path="configuration", config_name="config_test", version_base=None)
def main(cfg: DictConfig):
    # Инициализация задачи
    param_bounds = cfg.parameters
    objectives = cfg.objectives
    bb_function = BlackBoxFunction(param_bounds, objectives)

    param_keys = list(param_bounds.keys())
    dim = len(param_keys)

    # Начальная выборка (LHS или случайная)
    X_init = latin_hypercube_sampling(cfg.initial_samples, dim)
    Y_init = bb_function.evaluate(X_init)

    # Задача MOBO — минимизация по нескольким целевым функциям
    # Здесь добавьте реализацию через PHYSBO или свой подход
    X = X_init
    Y = Y_init

    for iteration in range(cfg.n_iter):
        # Использовать модель для оценки acquisition-функции на кандидатах
        X_candidates = latin_hypercube_sampling(cfg.num_candidates, dim)

        # Здесь должна быть функция выбора следующих точек (например, EHVI)
        # Ниже — просто заглушка (случайный выбор)
        chosen_idx = np.random.choice(cfg.num_candidates, cfg.batch_size, replace=False)
        X_new = X_candidates[chosen_idx]

        Y_new = bb_function.evaluate(X_new)

        # Обновление базы
        X = np.vstack([X, X_new])
        Y = np.vstack([Y, Y_new])

        print(f"[Iter {iteration+1}] Best Pareto estimate (min values): {np.min(Y, axis=0)}")

    # Построение фронта Парето (по всем точкам)
    evaluated_indices = np.unique(chosen_idx)
    X_evaluated = X_candidates[evaluated_indices]
    Y_evaluated = bb_function.evaluate(X_evaluated)

    # Опционально: визуализация фронта Парето
    import matplotlib.pyplot as plt
    def is_dominated(y, Y):
        """Проверяет, доминируется ли точка y множеством Y (предполагается минимизация)."""
        return np.any(np.all(Y <= y, axis=1) & np.any(Y < y, axis=1))

    def get_pareto_front(Y):
        """Возвращает индексы недоминируемых точек (фронт Парето)."""
        pareto_indices = []
        for i in range(Y.shape[0]):
            if not is_dominated(Y[i], np.delete(Y, i, axis=0)):
                pareto_indices.append(i)
        return np.array(pareto_indices)

    pareto_idx = get_pareto_front(Y_evaluated)

    plt.figure(figsize=(6, 5))
    plt.scatter(Y_evaluated[:, 0], Y_evaluated[:, 1], c='blue', label='Evaluated')
    plt.scatter(Y_evaluated[pareto_idx, 0], Y_evaluated[pareto_idx, 1], c='red', label='Pareto front')
    plt.xlabel(objectives[0])
    plt.ylabel(objectives[1])
    plt.legend()
    plt.title("Pareto Front Approximation")
    plt.tight_layout()
    plt.show()
    print("\nOptimization finished.")
    # Здесь можно построить и сохранить фронт Парето по Y

if __name__ == "__main__":
    main()
