import tensorflow as tf
from classifier import BinaryClassifier
from os import listdir
import tflearn_classifier

POSITIVE = [1, 0]
NEGATIVE = [0, 1]

AUDIO_FEATURE_SIZE = 1280
EVAL_SET_SIZE = 100

def main():
    x = tf.placeholder(tf.float32, [None, AUDIO_FEATURE_SIZE])
    y = tf.placeholder(tf.float32, [None, 2])

    n_units = [100, 100, 50]
    n_batches = 10000
    batch_size = 10
    sound_event = 137 # Testing with sound event class of music.

    classifier = BinaryClassifier(x, y, n_units)

    path = "./trainingFeatures/bal_train/"
    filenames = [path + f for f in listdir(path)]
    """filenames = [path + "ZZ.tfrecord",
                 path + "Zy.tfrecord",
                 path + "ZY.tfrecord",
                 path + "zz.tfrecord",
                 path + "zZ.tfrecord",
                 path + "uT.tfrecord",
                 path + "Ut.tfrecord",
                 path + "UT.tfrecord",
                 path + "uu.tfrecord",
                 path + "uU.tfrecord"
                 ]"""
    eval_path = "./trainingFeatures/eval/"
    eval_filenames = [path + f for f in listdir(path)]

    batch = tf.train.batch(
        extract_example(filenames), batch_size, dynamic_pad = True)
    eval_batch = tf.train.batch(
            extract_example(eval_filenames), EVAL_SET_SIZE, dynamic_pad = True)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coordinator)
         
        # Train the model.
        for i in range(n_batches):
            labels, audio_features = sess.run(batch)
            targets = []

            for j in range(batch_size):
                if sound_event in labels[j]:
                    targets.append(POSITIVE)
                else:
                    targets.append(NEGATIVE)

            sess.run(classifier.train, 
                    feed_dict = {x: audio_features, y: targets})
        
        # Evaluate the model.
        labels, audio_features = sess.run(eval_batch)
        targets = []

        for i in range(EVAL_SET_SIZE):
            if sound_event in labels[i]:
                targets.append(POSITIVE)
            else:
                targets.append(NEGATIVE)

        print(sess.run(classifier.f1_score, 
            feed_dict = {x: audio_features, y: targets}))

        coordinator.request_stop()
        coordinator.join()

def extract_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    _, serialized_example = tf.TFRecordReader().read(filename_queue)

    data = tf.parse_single_sequence_example(
        serialized_example,
        {"labels": tf.VarLenFeature(tf.int64)},
        {"audio_embedding": tf.FixedLenSequenceFeature([], tf.string)})

    labels = tf.sparse_tensor_to_dense(data[0]["labels"])
    audio_features = tf.reshape(tf.cast(tf.decode_raw(
        data[1]["audio_embedding"], tf.uint8), tf.float32), [-1])

    return labels, audio_features

if __name__ == "__main__":
    #main()
    tflearn_classifier.classify()
