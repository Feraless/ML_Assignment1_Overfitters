import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics.py import mae,rmse,accuracy,precision,recall
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100


# Function to create fake data (take inspiration from usage.py)
def generate_data(N, M, input_type, output_type):
    if input_type == "real" and output_type == "real":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        return X, y
    elif input_type == "real" and output_type == "discrete":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size=N), dtype="category")
        return X, y
    elif input_type == "discrete" and output_type == "discrete":
        X = pd.DataFrame({i: pd.Series(np.random.randint(M, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(M, size=N), dtype="category")
        return X, y
    elif input_type == "discrete" and output_type == "real":
        X = pd.DataFrame({i: pd.Series(np.random.randint(M, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
        return X, y


# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def measure_time(X_train, y_train, X_test, y_test):
    tree = DecisionTree(criterion="entropy")

    # Measure training time
    start_time = time.time()
    node = tree.fit(X_train, y_train, criterion="entropy")
    train_time = time.time() - start_time

    # Measure prediction time
    start_time = time.time()
    predictions = tree.predict(X_test, node)
    predict_time = time.time() - start_time

    return train_time, predict_time


# Function to plot time complexity graphs
def plot_time_complexity(N_values, M_values):
    scenarios = [
        ("discrete", "discrete"),
        ("discrete", "real"),
        ("real", "discrete"),
        ("real", "real")
    ]
    scenario_labels = ["Discrete-Discrete", "Discrete-Real", "Real-Discrete", "Real-Real"]

    fig, axs = plt.subplots(len(M_values), 2, figsize=(15, 20))
    fig.suptitle("Time vs Number of Features (N) - Criteria: information_gain")

    for j, M in enumerate(M_values):
        train_times = {label: [] for label in scenario_labels}
        predict_times = {label: [] for label in scenario_labels}

        for N in N_values:
            for scenario, label in zip(scenarios, scenario_labels):
                X, y = generate_data(N, M, scenario[0], scenario[1])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                avg_train_time, avg_predict_time = 0, 0

                # Averaging over multiple runs
                train_time, predict_time = measure_time(X_train, y_train, X_test, y_test)
                avg_train_time += train_time
                avg_predict_time += predict_time

                # avg_train_time /= num_average_time
                # avg_predict_time /= num_average_time

                train_times[label].append(avg_train_time)
                predict_times[label].append(avg_predict_time)

        # Plot training time vs N
        for label in scenario_labels:
            axs[j, 0].plot(N_values, train_times[label], label=label, marker='o')
        axs[j, 0].set_title(f'Training Time vs Number of Samples (N), M = {M}')
        axs[j, 0].set_xlabel('Number of Samples (N)')
        axs[j, 0].set_ylabel('Training Time (seconds)')
        axs[j, 0].grid(True)
        axs[j, 0].legend()

        # Plot prediction time vs N
        for label in scenario_labels:
            axs[j, 1].plot(N_values, predict_times[label], label=label, marker='o')
        axs[j, 1].set_title(f'Prediction Time vs Number of Samples (N), M = {M}')
        axs[j, 1].set_xlabel('Number of Samples (N)')
        axs[j, 1].set_ylabel('Prediction Time (seconds)')
        axs[j, 1].grid(True)
        axs[j, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    fig, axs = plt.subplots(len(N_values), 2, figsize=(15, 20))
    fig.suptitle("Time vs Number of Features (M) - Criteria: information_gain")

    for i, N in enumerate(N_values):
        train_times = {label: [] for label in scenario_labels}
        predict_times = {label: [] for label in scenario_labels}

        for M in M_values:
            for scenario, label in zip(scenarios, scenario_labels):
                X, y = generate_data(N, M, scenario[0], scenario[1])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                avg_train_time, avg_predict_time = 0, 0

                # Averaging over multiple runs
                train_time, predict_time = measure_time(X_train, y_train, X_test, y_test)
                avg_train_time += train_time
                avg_predict_time += predict_time

                # avg_train_time /= num_average_time
                # avg_predict_time /= num_average_time

                train_times[label].append(avg_train_time)
                predict_times[label].append(avg_predict_time)

        # Plot training time vs M
        for label in scenario_labels:
            axs[i, 0].plot(M_values, train_times[label], label=label, marker='o')
        axs[i, 0].set_title(f'Training Time vs Number of Features (M), N = {N}')
        axs[i, 0].set_xlabel('Number of Features (M)')
        axs[i, 0].set_ylabel('Training Time (seconds)')
        axs[i, 0].grid(True)
        axs[i, 0].legend()

        # Plot prediction time vs M
        for label in scenario_labels:
            axs[i, 1].plot(M_values, predict_times[label], label=label, marker='o')
        axs[i, 1].set_title(f'Prediction Time vs Number of Features (M), N = {N}')
        axs[i, 1].set_xlabel('Number of Features (M)')
        axs[i, 1].set_ylabel('Prediction Time (seconds)')
        axs[i, 1].grid(True)
        axs[i, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    fig, axs = plt.subplots(len(N_values), 2, figsize=(15, 20))
    fig.suptitle("Time vs Number of Features (M) - Criteria: information_gain")

    for i, N in enumerate(N_values):
        train_times = {label: [] for label in scenario_labels}
        predict_times = {label: [] for label in scenario_labels}

        for M in M_values:
            for scenario, label in zip(scenarios, scenario_labels):
                X, y = generate_data(N, M, scenario[0], scenario[1])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                avg_train_time, avg_predict_time = 0, 0

                # Averaging over multiple runs
                train_time, predict_time = measure_time(X_train, y_train, X_test, y_test)
                avg_train_time += train_time
                avg_predict_time += predict_time

                # avg_train_time /= num_average_time
                # avg_predict_time /= num_average_time

                train_times[label].append(avg_train_time)
                predict_times[label].append(avg_predict_time)

        # Plot training time vs M
        for label in scenario_labels:
            axs[i, 0].plot(M_values, train_times[label], label=label, marker='o')
        axs[i, 0].set_title(f'Training Time vs Number of Features (M), N = {N}')
        axs[i, 0].set_xlabel('Number of Features (M)')
        axs[i, 0].set_ylabel('Training Time (seconds)')
        axs[i, 0].grid(True)
        axs[i, 0].legend()

        # Plot prediction time vs M
        for label in scenario_labels:
            axs[i, 1].plot(M_values, predict_times[label], label=label, marker='o')
        axs[i, 1].set_title(f'Prediction Time vs Number of Features (M), N = {N}')
        axs[i, 1].set_xlabel('Number of Features (M)')
        axs[i, 1].set_ylabel('Prediction Time (seconds)')
        axs[i, 1].grid(True)
        axs[i, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


# Example usage
N_values = [2,5,10,20,50]  # Different values of N
M_values = [5,10, 20, 50]  # Different values of M
plot_time_complexity(N_values, M_values)
