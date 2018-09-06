import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def create_output_dir():
    """Create output dir if it does not exist."""
    cwd = os.getcwd()
    output_dir_path = os.path.join(cwd, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

def generate_linear_data(num_data_points):
    X = np.random.rand(num_data_points, 2)

    # Set classes with the line x_2 = 0.5 as a boundary
    Y = np.expand_dims([0 if (X[i, 1] > 0.5) else 1 for i in range(num_data_points)], axis=1)

    # Flip the class of the points randomly, with higher probability if the point is close to the boundary
    for i in range(Y.shape[0]):
        distance = abs(X[i, 1] - 0.5)
        flip_prob = max(0, 0.3 - distance)
        if np.random.rand() < flip_prob:
            Y[i] = (Y[i] + 1) % 2

    return X, Y

def generate_non_linear_data(num_data_points):
    X = np.random.rand(num_data_points, 2)

    # Set classes with a ring centered at (0.5, 0.5) as a boundary
    Y = np.expand_dims([0 if (np.linalg.norm(X[i] - np.array([0.5, 0.5])) < 0.3) else 1 for i in range(num_data_points)], axis=1)

    return X, Y

def plot_data(X, Y, plot_name):
    colors = Y[:, 0]
    plt.scatter(X[:, 0], X[:, 1], c=colors, cmap=mpl.colors.ListedColormap(["red", "blue"]), edgecolors=["black"])
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def plot_decision_boundary(X_train, Y_train, predict_func, plot_name):
    interval = np.arange(-0.1, 1.1, 0.001)
    X_1, X_2 = np.meshgrid(interval, interval)
    X_grid = np.c_[X_1.ravel(), X_2.ravel()]
    Y_grid = predict_func(X_grid)
    predictions_grid = np.array([round(x[0]) for x in Y_grid])
    predictions_grid = predictions_grid.reshape(X_1.shape)
    plt.contourf(X_1, X_2, predictions_grid, cmap=mpl.colors.ListedColormap(["#ff6868", "#6875ff"]))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0], cmap=mpl.colors.ListedColormap(["red", "blue"]), edgecolors=["black"])
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()