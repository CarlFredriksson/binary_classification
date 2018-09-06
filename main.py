import bc_utils
from logistic_regression import logistic_regression_2D
from nn_classification import nn_binary_classification_2D

LEARNING_RATE = 0.1
NUM_ITERATIONS = 100000

bc_utils.create_output_dir()
results_file = open("output/results.txt", "w")

# Linear data
X_train, Y_train = bc_utils.generate_linear_data(300)
bc_utils.plot_data(X_train, Y_train, "data_linear_train.png")
X_test, Y_test = bc_utils.generate_linear_data(300)
bc_utils.plot_data(X_test, Y_test, "data_linear_test.png")

J_train, J_test = logistic_regression_2D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS, "lr_db_linear_train.png")
results_file.write("Logistic Regression - linear data> J_train: " + str(J_train) + ", J_test: " + str(J_test) + "\n")

J_train, J_test = nn_binary_classification_2D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS, "nn_db_linear_train.png")
results_file.write("NN Classification - linear data> J_train: " + str(J_train) + ", J_test: " + str(J_test) + "\n")

# Non-linear data
X_train, Y_train = bc_utils.generate_non_linear_data(300)
bc_utils.plot_data(X_train, Y_train, "data_non_linear_train.png")
X_test, Y_test = bc_utils.generate_non_linear_data(300)
bc_utils.plot_data(X_test, Y_test, "data_non_linear_test.png")

J_train, J_test = logistic_regression_2D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS, "lr_db_non_linear_train.png")
results_file.write("Logistic Regression - non-linear data> J_train: " + str(J_train) + ", J_test: " + str(J_test) + "\n")

J_train, J_test = nn_binary_classification_2D(X_train, Y_train, X_test, Y_test, LEARNING_RATE, NUM_ITERATIONS,  "nn_db_non_linear_train.png")
results_file.write("NN Classification - non-linear data> J_train: " + str(J_train) + ", J_test: " + str(J_test) + "\n")
