import numpy as np
import physbo
import pandas as pd


# Класс черного ящика с несколькими целевыми функциями
class BlackBoxFunction:
    def __init__(self, param_bounds, objectives):
        self.param_bounds = param_bounds
        self.objectives = objectives

    def scale_parameters(self, X_scaled):
        """
        Преобразует параметры из нормализованного вида [0,1] в оригинальные диапазоны.
        """
        X_original = np.zeros_like(X_scaled)
        for idx, key in enumerate(self.param_bounds.keys()):
            min_val, max_val = self.param_bounds[key]
            X_original[:, idx] = X_scaled[:, idx] * (max_val - min_val) + min_val
        return X_original

    def evaluate(self, X_scaled):
        """
        Вычисляет значения целевых функций.
        X_scaled: (n_samples, n_params) нормализованные параметры от 0 до 1.
        Возвращает массив значений размерности (n_samples, n_objectives).
        """
        X = self.scale_parameters(X_scaled)

        # Пример синтетических функций (заменить на собственные функции)
        obj1 = np.sum(X ** 2, axis=1)  # Минимизировать сумму квадратов
        obj2 = np.sum((X - 2) ** 2, axis=1)  # Минимизировать сумму квадратов относительно 2

        # Возвращает значения в виде DataFrame для удобства
        return pd.DataFrame(np.vstack([obj1, obj2]).T, columns=self.objectives)


if __name__ == '__main__':
    # Задание диапазонов параметров
    parameters = {
        'param1': (-5, 5),
        'param2': (0, 10),
        'param3': (1, 6),
        'param4': (-2, 3),
        'param5': (0, 1),
    }

    # Список целевых функций
    objectives = ['objective1', 'objective2']

    # Создание экземпляра функции
    blackbox = BlackBoxFunction(parameters, objectives)

    # Генерация пространства поиска (X_scaled от 0 до 1)
    num_candidates = 2000
    X_scaled = np.random.rand(num_candidates, len(parameters))

    # Инициализация оптимизатора physbo
    optimizer = physbo.search.discrete.policy(test_X=X_scaled)

    # Первоначальная случайная выборка
    np.random.seed(0)
    initial_samples = 15
    optimizer.random_sampling(max_num_probes=initial_samples, simulator=lambda X: blackbox.evaluate(X).values[:, 0])

    # Цикл байесовской оптимизации (используем первую целевую функцию для упрощения)
    n_iter = 30
    for i in range(n_iter):
        optimizer.bayes_search(max_num_probes=1,
                               simulator=lambda X: blackbox.evaluate(X).values[:, 0],
                               score='EI')

    # Получение оптимального решения по первой целевой функции
    best_sample_index = optimizer.history.fx.argmin()
    best_x_scaled = X_scaled[best_sample_index]
    best_x_original = blackbox.scale_parameters(best_x_scaled.reshape(1, -1))
    best_objective_values = blackbox.evaluate(best_x_scaled.reshape(1, -1))

    print("Оптимальные параметры:")
    for idx, key in enumerate(parameters.keys()):
        print(f"{key}: {best_x_original[0, idx]}")

    print("\nЗначения целевых функций в оптимуме:")
    print(best_objective_values)
