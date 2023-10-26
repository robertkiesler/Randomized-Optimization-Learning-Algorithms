import mlrose_hiive as mlrose
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import preprocessing
import time

def load_data(dataset):
    df = pd.read_csv("data/parkinsons.data")
    X = df.drop(['status', 'name'], axis=1)
    y = df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = preprocessing.scale(X_train)
    X_test = preprocessing.scale(X_test)

    return X_train, X_test, y_train, y_test

def train_neural_network(algorithm, train_x, train_y, test_x, test_y, max_iters=1500, max_attempts=50,
                         early_stopping=False, mutation_prob=0.05, hidden_nodes=[32, 32], pop_size=200, restarts=75):
    nn_model = mlrose.NeuralNetwork(hidden_nodes=hidden_nodes, activation='relu', \
                                    algorithm=algorithm, max_iters=max_iters, \
                                    bias=True, is_classifier=True, learning_rate=0.001,
                                    clip_max=5, max_attempts=max_attempts, random_state=10, curve=True,
                                    early_stopping=early_stopping, mutation_prob=mutation_prob, pop_size=pop_size, restarts=restarts)

    start = time.time()
    nn_model.fit(train_x, train_y)
    end = time.time()
    time_elapsed = end - start

    y_train_pred = nn_model.predict(train_x)
    y_test_pred = nn_model.predict(test_x)
    y_test_accuracy = accuracy_score(test_y, y_test_pred)

    return {
        'model': nn_model,
        'time_elapsed': time_elapsed,
        'train_accuracy': accuracy_score(train_y, y_train_pred),
        'test_accuracy': y_test_accuracy
    }

def iterative_train(times, accuracies, iters_range, algorithms, train_x, train_y, test_x, test_y,
                    early_stopping=False, mutation_prob=0.05):
    times_dict = {}
    accuracies_dict = {}

    for algorithm in algorithms:
        algorithm_times = []
        algorithm_accuracies = []

        for i in iters_range:
            result = train_neural_network(algorithm, train_x, train_y, test_x, test_y, i,
                                          early_stopping=early_stopping, mutation_prob=mutation_prob)

            algorithm_times.append(result['time_elapsed'])
            algorithm_accuracies.append(result['test_accuracy'])

        if len(algorithm_times) != len(iters_range) or len(algorithm_accuracies) != len(iters_range):
            raise ValueError("Mismatch in the lengths of algorithm_times and iters_range")

        times_dict[algorithm] = algorithm_times
        accuracies_dict[algorithm] = algorithm_accuracies

    return times_dict, accuracies_dict

def plot_graph(x_values, y_values_dict, title, x_label, y_label, legend_labels, save_filename):
    plt.figure()
    for label, y_values in y_values_dict.items():
        plt.plot(x_values, y_values, 'o-', label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend_labels)
    plt.savefig(save_filename)
    plt.clf()

if __name__ == "__main__":
    nn_train_x, nn_test_x, nn_train_y, nn_test_y = load_data("parkinsons.data")

    iters_range = [10, 50, 100, 250, 500]
    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg', 'gradient_descent']

    times_dict, accuracies_dict = iterative_train([], [], iters_range, algorithms, nn_train_x, nn_train_y, nn_test_x, nn_test_y, early_stopping=False, mutation_prob=0.1)

    plot_graph(iters_range, times_dict, "NN Time and Iterations", "Number of Iterations", "Time", algorithms, "graphs/training_time_combined.png")
    plot_graph(iters_range, accuracies_dict, "NN Accuracy and Iterations", "Number of Iterations", "Accuracy", algorithms, "graphs/test_accuracy_combined.png")

    print("\n\n------- Accuracy Report -------")
    for algo, accuracies in accuracies_dict.items():
        max_accuracy = max(accuracies)
        print(f"{algo}: Highest Test Accuracy = {max_accuracy:.4f}")
