"""Main python file for WITS project"""
import tensorflow as tf
import tflearn_classifier
import data
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import tflearn
def main():
    """Main function"""
    #tflearn_classifier.classify()    
    extract_test()


def extract_test():
    features = data.extract_example("./generatedFeatures/test.tfrecord")
    print(features)

main()