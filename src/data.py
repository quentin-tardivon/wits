"""This module is to open the tensorflow data"""
import tensorflow as tf

def _parse_function(example_proto):
  features = {"video_id": tf.FixedLenFeature((), tf.string),
              "start_time_seconds": tf.FixedLenFeature((), tf.float32),
              "end_time_seconds": tf.FixedLenFeature((), tf.float32)
              }
  parsed_features = tf.parse_single_example(example_proto, features)
  return parsed_features["video_id"], parsed_features["start_time_seconds"],  parsed_features["end_time_seconds"]


def extract_data():
    sess = tf.Session()
    """Function to extract the data form tfrecord"""
    filename = "features/eval/00.tfrecord"
    dataset = tf.data.TFRecordDataset(filename)
    print(dataset)
    dataset = dataset.map(_parse_function)
    print(dataset)
    ite = dataset.make_one_shot_iterator()
    next_element = ite.get_next()
    print(sess.run(next_element))
