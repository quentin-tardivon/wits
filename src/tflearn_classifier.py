from os import listdir
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data
import tflearn

def classify():
    sess = tf.Session()
    path = "./trainingFeatures/bal_train/"
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
        if len(features[i]) >= 10:
            for j in range(10):
                for value in features[i][j]:
                    temp.append(value)
            finalFeatures.append(np.array(temp))
            finalLabels.append(labels[i])
    
    finalFeatures = np.array(finalFeatures)
    finalLabels = np.array(finalLabels)
    #plt.scatter(indiceTab, sumTab, c=colors)
    #plt.show()

    X = finalFeatures
    outputs = np.zeros((len(finalLabels), 2))
    i = 0
    for labels in finalLabels:
        for label in labels:
            if label == 0:
                outputs[i][0] = 1
                break
            else:
                outputs[i][1] = 1
        i += 1
    print(outputs)
    Y = outputs
    net = tflearn.input_data(shape=[None, 1280])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net, tensorboard_verbose=3)
    model.fit(X, Y, n_epoch=2,validation_set=0.3,
             show_metric=True, batch_size=64, snapshot_step=10)
    
    
    path = "./trainingFeatures/eval/"
    filenames = [path + f for f in listdir(path)]
    features, labels = data.extract_data(filenames)
    finalFeatures = []
    finalLabels = []
    for i in range(len(features)):
        temp = []
        if len(features[i]) >= 10:
            for j in range(10):
                for value in features[i][j]:
                    temp.append(value)
            finalFeatures.append(np.array(temp))
            finalLabels.append(labels[i])
    i = 0
    for labels in finalLabels:
        for label in labels:
            if label == 0:
                outputs[i][0] = 1
            else:
                outputs[i][1] = 1
        i += 1
    predict = model.predict(finalFeatures)
    for i in range(len(predict)):
        if predict[i][0] > predict[i][1]:
            predict[i][0] = 1
            predict[i][1] = 0
        else:
            predict[i][1] = 0
            predict[i][0] = 1
    totalTrue = 0
    total = 0
    for i in range(len(predict)):
        if (outputs[i][0] == predict[i][0]):
            totalTrue += 1
        total += 1
    print(totalTrue/total)