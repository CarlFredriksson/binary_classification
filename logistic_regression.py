import numpy as np
import tensorflow as tf
import bc_utils

def logistic_regression_2D(X_train, Y_train, X_test, Y_test, learning_rate, num_iterations, db_plot_name):
    tf.reset_default_graph()
    
    # Create parameters
    W = tf.get_variable("W", shape=(2, 1), initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", shape=(1, 1), initializer=tf.zeros_initializer())

    # Compute cost
    X_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="X_placeholder")
    Y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="Y_placeholder")
    Y_predict = tf.sigmoid(tf.matmul(X_placeholder, W) + b)
    J = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_placeholder, logits=Y_predict))

    # Create train op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(J)

    # Start session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Training loop
        for i in range(num_iterations):
            sess.run(train_op, feed_dict={X_placeholder: X_train, Y_placeholder: Y_train})
            J_train = sess.run(J, feed_dict={X_placeholder: X_train, Y_placeholder: Y_train})
            if i%1000 == 0:
                print("i: " + str(i) + ", J_train: " + str(J_train))

        # Evaluate
        J_train = sess.run(J, feed_dict={X_placeholder: X_train, Y_placeholder: Y_train})
        J_test = sess.run(J, feed_dict={X_placeholder: X_test, Y_placeholder: Y_test})

        # Plot decision boundary
        predict_func = lambda X_g: sess.run(Y_predict, feed_dict={X_placeholder: X_g, Y_placeholder: Y_train})
        bc_utils.plot_decision_boundary(X_train, Y_train, predict_func, db_plot_name)

        return J_train, J_test
