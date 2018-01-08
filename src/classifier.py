import tensorflow as tf
import functools

def lazy_property(function):
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class BinaryClassifier:   
    def __init__(self, audio_features, labels, n_units):
        self.audio_features = audio_features
        self.labels = labels
        self.n_units = n_units
        self.predictions
        self.train
        self.f1_score
       
    @lazy_property
    def predictions(self):
        n_layers = len(self.n_units)
        x = self.audio_features
        for i in range(n_layers):
            x = tf.contrib.layers.fully_connected(
                x, self.n_units[i], tf.nn.sigmoid)
        return tf.contrib.layers.fully_connected(x, 2, None)

    @lazy_property
    def train(self): 
        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels = self.labels, logits = self.predictions))
        return tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    @lazy_property
    def f1_score(self):
        sensitivity = tf.metrics.recall(
                self.labels, tf.round(tf.nn.softmax(self.predictions)))[1]
        precision = tf.metrics.precision(
             self.labels, tf.round(tf.nn.softmax(self.predictions)))[1]
        return 2 * precision * sensitivity / (precision + sensitivity)
