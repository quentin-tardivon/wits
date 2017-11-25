"""Main python file for WITS project"""
import tensorflow as tf
import data
def main():
    """Main function"""
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))
    data.extract_data()

main()
