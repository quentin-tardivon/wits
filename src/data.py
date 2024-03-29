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


def extract_data(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    features = []
    labels = []
    while True:
        try:
            labels.append(sess.run(next_element[1]))
            features.append(sess.run(next_element[2]))
        except tf.errors.OutOfRangeError:
            break
    return features, labels

def extract_generated_data(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser_generated)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.Session()
    features = []
    while True:
        try:
            features.append(sess.run(next_element[0]))
        except tf.errors.OutOfRangeError:
            break
    return features

def parser(sequence_example):
    context_features = {
            "video_id": tf.FixedLenFeature([], tf.string),
            "labels": tf.VarLenFeature(tf.int64)
            }
    sequence_features = {
            "audio_embedding": tf.FixedLenSequenceFeature([], tf.string)
            }   
     
    contexts, sequences = tf.parse_single_sequence_example(sequence_example, context_features, sequence_features)
    video_ids, labels = contexts["video_id"], tf.sparse_tensor_to_dense(contexts["labels"])
    audio_features = tf.decode_raw(sequences["audio_embedding"], tf.uint8)
    return video_ids, labels, audio_features


def parser_generated(sequence_example):
    sequence_features = {
            "audio_embedding": tf.FixedLenSequenceFeature([], tf.string)
            }   
     
    sequences = tf.parse_single_sequence_example(sequence_example, sequence_features)
    audio_features = tf.decode_raw(sequences["audio_embredding"], tf.uint8)
    return audio_features
