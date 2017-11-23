# Feel free to add any functions, import statements, and variables.
import numpy as np
import pandas as pd
import tensorflow as tf

def predict(file):
    # Fill in this function. This function should return a list of length 52
    #   which is filled with floating point numbers. For example, the current
    #   implementation predicts all the instances in test.csv as 10.0.

    # Get train dataset (raw dataset with features and labels)
    train_df = pd.read_csv('train.csv', engine='python')
    # Will skip validation
    Xtr = train_df.iloc[:, 1:-1].values.reshape(-1, 6, 1).astype("float")
    Ytr = train_df.iloc[:, -1].values.reshape(-1, 1).astype("float")

    # Set parameters
    seq_length = 6
    data_dim = 1 # dimension
    num_classes = 1

    with tf.device('/gpu:0'):
        X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        Y = tf.placeholder(tf.float32, [None, num_classes])
    
    # Define a LSTM cell with TensorFlow
    num_hidden = 512 # Hidden layer number of features

    cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0,
                                        state_is_tuple=True, activation=tf.tanh)
    # Get cell output
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    
    # Use the last cell output
    Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=None)
    
    # Define loss and optimizer
    learning_rate = 0.001

    # cost/loss
    loss_op = tf.reduce_sum(tf.square(Y_pred - Y)) # Sum of squares
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Initialize the variables (i.e. assign their default value)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Train
    training_steps = 500

    for step in range(1, training_steps + 1):
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: Xtr, Y: Ytr})

    # Test
    test_df = pd.read_csv(file, engine='python')
    # Get Xte in ndarrays
    # Y for testing is not given
    # From F1 to F6 as matrix 6X1
    Xte = test_df.iloc[:, 1:].values.reshape(-1, 6, 1).astype("float")
    testPredict = sess.run(Y_pred, feed_dict={X: Xte})
    predictions = testPredict.flatten()

    return [pred for pred in predictions]


def write_result(predictions):
    # You don't need to modify this function.
    with open('result.csv', 'w') as f:
        f.write('Value\n')
        for l in predictions:
            f.write('{}\n'.format(l))


def main():
    # You don't need to modify this function.
    predictions = predict('test.csv')
    write_result(predictions)


if __name__ == '__main__':
    # You don't need to modify this part.
    main()
