#         numpy_cross_entropy -- Min: 0.000089s, Mean: 0.000112s, Std Dev: 0.000021s
# categorical_cross_entropy_1 -- Min: 0.000424s, Mean: 0.000512s, Std Dev: 0.000107s
# categorical_cross_entropy_2 -- Min: 0.000415s, Mean: 0.000493s, Std Dev: 0.000077s
# categorical_cross_entropy_3 -- Min: 0.000019s, Mean: 0.000024s, Std Dev: 0.000004s
# categorical_cross_entropy_4 -- Min: 0.000010s, Mean: 0.000013s, Std Dev: 0.000003s

import timeit
import random
import numpy as np

import math

def numpy_cross_entropy(y_actual, y_predicted):
    loss = -np.sum(y_actual * np.log(y_predicted))
    return loss

def categorical_cross_entropy_1(y_actual, y_predicted):
    y_pred = [min(max(predicted, 1e-15), 1 - 1e-15) for predicted in y_predicted]
    return - sum(true_i * math.log(pred_i) for true_i, pred_i in zip(y_actual, y_pred))

def categorical_cross_entropy_2(y_actual, y_predicted):
    loss = 0
    for y, ŷ in zip(y_actual, y_predicted):
        ŷ = min(max(ŷ, 1e-15), 1 - 1e-15)
        loss += y * math.log(ŷ)
    return -loss

def categorical_cross_entropy_3(y_actual, y_predicted):
    '''assumes proper one-hot encoding, short circuits when true class is found'''
    for y, ŷ in zip(y_actual, y_predicted):
        if y == 1:
            ŷ = min(max(ŷ, 1e-15), 1 - 1e-15)
            return -math.log(ŷ)
    raise ValueError("y_actual is not properly one-hot encoded (no '1' found).")

def categorical_cross_entropy_4(y_actual, y_predicted):
    '''return the '''
    try:
        y_i = y_actual.index(1)
    except ValueError:
        raise ValueError("y_actual is not properly one-hot encoded (no '1' found).")
    clipped_prediction = min(max(y_predicted[y_i], 1e-15), 1 - 1e-15)
    return -math.log(clipped_prediction)

functions = [numpy_cross_entropy, categorical_cross_entropy_1, categorical_cross_entropy_2, categorical_cross_entropy_3, categorical_cross_entropy_4]
list_length = 1_000
num_runs = 100_000
arg1 = list(np.zeros(list_length))
index = random.randint(0, list_length-1)
print(f'{index=}')
arg1[index] = 1
arg2 = [random.random() for _ in range(list_length)]

def benchmark(function, arg1, arg2, num_runs):
    setup_code = f"from __main__ import {function.__name__}, arg1, arg2"
    test_code = f"{function.__name__}(arg1, arg2)"
    times = timeit.repeat(setup=setup_code, stmt=test_code, repeat=num_runs, number=1)
    return min(times), np.mean(times), np.std(times)

for f in functions:
    times = benchmark(f, arg1, arg2, num_runs)
    print(f"{f.__name__} -- Min: {times[0]:.6f}s, Mean: {times[1]:.6f}s, Std Dev: {times[2]:.6f}s")