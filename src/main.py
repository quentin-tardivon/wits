"""Main python file for WITS project"""
import tensorflow as tf
import data
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import tflearn
def main():
    """Main function"""
    sess = tf.Session()
    path = "../trainingFeatures/bal_train/"
    filenames = [path + f for f in listdir(path)]
    features, labels = data.extract_data(filenames)
    features = np.array(features)
    total = 0
    i = 0
    sumTab = np.zeros((3985))
    colors = np.zeros((3985))
    indiceTab = np.zeros((3985))
    finalFeatures = []
    finalLabels = []
    for f in features:
        for label in labels[i]:
            if label == 0:
                colors[i] = 1
        sumTab[i] = sum(sum(f))
        indiceTab[i] = i
        i+=1
    
    
    for i in range(len(features)):
        temp = []
        if len(features[i]) == 10:
            for f in features[i]:
            
                for value in f:
                    temp.append(value)
        
                #temp.append(f)
            finalFeatures.append(np.array(temp))
            finalLabels.append(labels[i])
    
    finalFeatures = np.array(finalFeatures)
    finalLabels = np.array(finalLabels)
    print(finalFeatures.shape)
    plt.scatter(indiceTab, sumTab, c=colors)
    plt.show()

    X = finalFeatures[0:3000]
    outputs = np.zeros((len(finalLabels), 2))
    i = 0
    for labels in finalLabels:
        for label in labels:
            if label == 0:
                outputs[0] = 1
            else:
                outputs[1] = 1
        i += 1
    Y = outputs[0:3000]
    testX = finalFeatures[3000::]
    testY = outputs[3000::]
    net = tflearn.input_data(shape=[None, 1280])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net, tensorboard_verbose=3)
    model.fit(X, Y, n_epoch=10,validation_set=(testX, testY),
             show_metric=True, batch_size=64, snapshot_step=10)

main()
