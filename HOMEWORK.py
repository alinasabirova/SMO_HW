import numpy as np
from scipy.optimize import minimize
import cec2017.functions as functions
import time
import pandas as pd
import pyswarms as ps
import pygad
import scikit_posthocs as sp
from scipy import stats

# дз1
import cec2017.utils as utils

function_names = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10']

for func_name in function_names:
    func = getattr(functions, func_name)
    utils.surface_plot(func, points=120)

# ДЗ №2
# Функция для оценки времени выполнения
def time_execution(func, x0):
    start_time = time.time()
    result = minimize(func, x0, method='BFGS')
    end_time = time.time()
    return result, end_time - start_time

# Функция-обертка для преобразования одномерного массива в двумерный
def wrapper_func(func):
    def wrapped_func(x):
        return func(np.array([x]))[0]
    return wrapped_func

# Создание списка функций для оптимизации
functions_list = [getattr(functions, f'f{i}') for i in range(1, 29) if i not in [17, 20]]

# Инициализация DataFrame для хранения результатов
results_df = pd.DataFrame(columns=[f'f{i}' for i in range(1, 29) if i not in [17, 20]], index=range(10))

# Запуск оптимизации для каждой функции 10 раз
for run in range(10):
    for func in functions_list:
        # Оборачиваем функцию, чтобы она принимала одномерный массив
        wrapped_func = wrapper_func(func)
        # Инициализация параметров
        x0 = 200 * np.random.random_sample((10,)) - 100
        result, execution_time = time_execution(wrapped_func, x0)
        # Сохранение результатов в DataFrame
        results_df.loc[run, func.__name__] = result.fun
results_df.to_excel("2.xlsx")

# ДЗ №3а

# Функция для оценки времени выполнения
def time_execution_pso(func):
    start_time = time.time()
    # Set-up hyperparameters
    options = {'c1': np.round(0.5*np.random.random()+0.25,2), 'c2': np.round(0.3*np.random.random()+0.1,2), 'w':0.9}
    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=10, options=options)
    # Perform optimization
    cost, pos = optimizer.optimize(func, iters=1000)
    end_time = time.time()
    return cost, end_time - start_time

# Создание списка функций, исключая f17 и f20
functions_list = [getattr(functions, f'f{i}') for i in range(1, 29) if i not in [17, 20]]

# Инициализация DataFrame для хранения результатов PSO
results_pso_df = pd.DataFrame(columns=[f'f{i}' for i in range(1, 29) if i not in [17, 20]], index=range(10))

# Запуск оптимизации для каждой функции 10 раз
for run in range(10):
    for func in functions_list:
        # Инициализация параметров
        cost, execution_time = time_execution_pso(func)
        # Сохранение результатов в DataFrame
        results_pso_df.loc[run, func.__name__] = cost
results_pso_df.to_excel("3a.xlsx")

# ДЗ №3б

def time_execution_ga(func):
    start_time = time.time()
    # Set-up hyperparameters
    num_generations = 50
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = 10
    init_range_low = -100
    init_range_high = 100
    parent_selection_type = "sss"
    keep_parents = 1
    crossover_type = "single_point"
    mutation_type = "random"
    mutation_percent_genes = 10

    # Fitness function
    def fitness_func(ga_instance, solution, solution_idx):
        # Преобразование одномерного массива в двумерный, если это необходимо
        if len(solution.shape) == 1:
            solution = solution.reshape(1, -1)
        return func(solution)

    # Create an instance of the GA class
    ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_func,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=init_range_low,
                               init_range_high=init_range_high,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes)

    # Run the GA
    ga_instance.run()

    # Get the best solution
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    end_time = time.time()
    return solution_fitness, end_time - start_time

results_ga_df = pd.DataFrame(columns=[f'f{i}' for i in range(1, 29) if i not in [17, 20]], index=range(10))

for run in range(10):
    for func in [getattr(functions, f'f{i}') for i in range(1, 29) if i not in [17, 20]]:
        cost, execution_time = time_execution_ga(func)
        results_ga_df.loc[run, func.__name__] = cost
results_ga_df.to_excel("3b.xlsx")

def is_less_than_0_05(value):
    return value < 0.05

results_summary = pd.DataFrame(columns=[f'f{i}' for i in range(1, 29) if i not in [17, 20]], index=['friedman', 'nemenyi'])

for i in range(1, 29):
    if i not in [17, 20]:
        f_minimize = results_df[f'f{i}'].values
        f_pso = results_pso_df[f'f{i}'].values
        f_ga = results_ga_df[f'f{i}'].values

        # Выполнение теста Фридмана
        friedman_result = stats.friedmanchisquare(f_minimize, f_pso, f_ga)

        # Пост-анализ Неменьи
        nemenyi_result = sp.posthoc_nemenyi_friedman(np.array([f_minimize, f_pso, f_ga]).T)

        friedman_sign = '+' if friedman_result[1] < 0.05 else '-'
        nemenyi_sign = '+' if (nemenyi_result < 0.05).any().any() else '-'

        results_summary.loc['friedman', f'f{i}'] = friedman_sign
        results_summary.loc['nemenyi', f'f{i}'] = nemenyi_sign

results_summary.to_excel("results4.xlsx")
