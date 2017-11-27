import tensorflow as tf

def extract_data():
    filenames = ["../features/bal_train/Xr.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parser)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    while True:
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            break

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
