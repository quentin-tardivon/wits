import tensorflow as tf

def extract_example(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    _, serialized_example = tf.TFRecordReader().read(filename_queue)

    features = tf.parse_single_sequence_example(
            serialized_example,
            {"labels": tf.VarLenFeature(tf.int64)},
            {"audio_embedding": tf.FixedLenSequenceFeature([], tf.string)})
    
    label = tf.sparse_tensor_to_dense(features[0]["labels"])
    audio_features = tf.reshape(tf.cast(tf.decode_raw(
        features[1]["audio_embedding"], tf.uint8), tf.float32), [-1])
    
    return label, audio_features
