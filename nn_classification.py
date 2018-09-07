import numpy as np
import tensorflow as tf
import bc_utils

def nn_binary_classification_2D(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, db_plot_name):
    tf.reset_default_graph()

    # Create parameters
    W_1 = tf.get_variable("W_1", shape=(2, 10), initializer=tf.contrib.layers.xavier_initializer())
    b_1 = tf.get_variable("b_1", shape=(1, 10), initializer=tf.zeros_initializer())

    W_2 = tf.get_variable("W_2", shape=(10, 10), initializer=tf.contrib.layers.xavier_initializer())
    b_2 = tf.get_variable("b_2", shape=(1, 10), initializer=tf.zeros_initializer())

    W_3 = tf.get_variable("W_3", shape=(10, 1), initializer=tf.contrib.layers.xavier_initializer())
    b_3 = tf.get_variable("b_3", shape=(1, 1), initializer=tf.zeros_initializer())

    # Forward propagation
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="X")
    Y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="Y")

    Y_hat = tf.matmul(X, W_1) + b_1
    Y_hat = tf.nn.relu(Y_hat)
    Y_hat = tf.matmul(Y_hat, W_2) + b_2
    Y_hat = tf.nn.relu(Y_hat)
    Y_hat = tf.sigmoid(tf.matmul(Y_hat, W_3) + b_3)

    # Compute cost, add small value epsilon to tf.log() calls to avoid taking the log of 0
    epsilon = 1e-10
    J = -tf.reduce_mean(Y * tf.log(Y_hat + epsilon) + (1 - Y) * tf.log(1 - Y_hat + epsilon))

    # Create train op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(J)

    # Start session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Training loop
        for i in range(num_epochs):
            sess.run(train_op, feed_dict={X: X_train, Y: Y_train})
            J_train = sess.run(J, feed_dict={X: X_train, Y: Y_train})
            if i%1000 == 0:
                print("i: " + str(i) + ", J_train: " + str(J_train))

        # Evaluate
        J_train = sess.run(J, feed_dict={X: X_train, Y: Y_train})
        J_test = sess.run(J, feed_dict={X: X_test, Y: Y_test})

        # Plot decision boundary
        predict_func = lambda X_grid: sess.run(Y_hat, feed_dict={X: X_grid, Y: Y_train})
        bc_utils.plot_decision_boundary(X_train, Y_train, predict_func, db_plot_name)

        return J_train, J_test
