import tensorflow as tf
from os import listdir
from data import extract_example
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram

SPEECH = 0

def create_and_train_model(
        filenames, num_units, batch_size, num_batches, learning_rate = 0.01):
    """
        Args:
            filenames: list of TFRecord filenames for files used in training.
            num_units: list of number of units in each layer.
            batch_size: size of each training batch.
            num_batches: number of batches until training ends.
    """

    num_layers = len(num_units)

    # Create the model
    x = tf.placeholder(tf.float32, [None, num_units[0]])
    y = tf.placeholder(tf.float32, [None, num_units[-1]])

    W, b = [], []

    for i in range(num_layers - 1):
        W.append(tf.Variable(
            tf.truncated_normal([num_units[i], num_units[i + 1]])))
        b.append(tf.Variable(tf.zeros([num_units[i + 1]])))

    q = tf.sigmoid(tf.matmul(x, W[0]) + b[0])

    for i in range(1, num_layers - 2):
        q = tf.sigmoid(tf.matmul(q, W[i]) + b[i])

    y_pred = tf.matmul(q, W[num_layers - 2]) + b[num_layers - 2]

    # Train the model
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels = y, logits = y_pred))
    optimizer = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(cost_function)

    labels_batch, features_batch = tf.train.batch(
            extract_example(filenames), batch_size, dynamic_pad = True)
    cost_history = np.empty(shape=[1],dtype=float)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coordinator)

        total, pos = (0, 0) # For demonstrating the problem.

        for i in range(num_batches):
            batch = sess.run([labels_batch, features_batch])
            if len(batch[1][0]) != 1280:
                # print(len(batch[1][0]))
                continue
            labels = []

            for j in range(batch_size):
                if SPEECH in batch[0][j]:
                    labels.append([1, 0])
                    pos = pos + 1
                else:
                    labels.append([0, 1])
                total = total + 1

            sess.run([optimizer,cost_function], feed_dict = {x: batch[1], y: labels})
            cost = sess.run(tf.nn.softmax(y_pred), feed_dict = {x: batch[1], y: labels})
            cost_history = np.append(cost_history,cost)
        
        print("% of positive examples: {:.1f} -- batch size: {}".format(
            100 * float(pos) / total, batch_size)) # For demonstrating the problem.
        coordinator.request_stop()
        coordinator.join(stop_grace_period_secs = 1)
        fig = plt.figure(figsize=(10,8))
        plt.plot(cost_history)
        plt.ylabel("Cost")
        plt.xlabel("Iterations")
        plt.axis([0,num_batches,0,np.max(cost_history)])
        plt.show()

# Uncomment if you wish to save the graph
#plt.savefig('training.pdf')
#plt.savefig('training.png')

# p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
# print("F-Score:")
# print(round(f,3))

"""
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: batch[1], y: labels}))
"""

if __name__ == "__main__":
    path = "../trainingFeatures/bal_train/"
    filenames = [path + f for f in listdir(path)]

    create_and_train_model(
            filenames, [1280, 600, 600, 2], 1, 1000) 