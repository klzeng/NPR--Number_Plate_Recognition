# coding: utf-8
import cv2
from neupy import algorithms, layers,plots
import pandas
import os
import preProcess as prePro
import featureExt as ftExt
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils


dict={'24': '4', '25': 'X', '26': '3', '27': 'E', '20': 'C', '21': 'D', '22': 'V',
      '23': 'Q', '28': 'B', '29': 'K', '1': 'U', '0': 'R', '3': '0', '2': '9', '5': 'I',
      '4': '7', '7': 'G', '6': 'N', '9': 'Z', '8': '6', '11': '8', '10': '1', '13': 'S', '12': 'T',
      '15': 'F', '14': 'A', '17': 'H', '16': 'O', '19': 'J', '18': 'M', '31': '2', '30': 'L', '35': 'W', '34': 'P', '33': '5', '32': 'Y'}

def get_data(datapath):
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for subdir, dir, files in os.walk(datapath):
        for file in files:
            if file.endswith('csv'):
                df = pandas.read_csv(datapath+'/'+file, header=None)
                datas = df.values
                X = datas[:,1:49]
                Y = datas[:,-1]
                len = X.shape[0]
                testcount = (int)(0.9*len)
                print "for " + file + " " + str(testcount) + "/" + str(len)
                for i in range(testcount):
                    x_train.append(X[i])
                    y_train.append(Y[i])
                for i in range(testcount,len):
                    x_test.append(X[i])
                    y_test.append(Y[i])

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    encoder.fit(y_test)
    y_test_encoded = encoder.transform(y_test)
    dummy_y_train = np_utils.to_categorical(y_train_encoded)
    dummy_y_test = np_utils.to_categorical(y_test_encoded)

    return np.array(x_train), dummy_y_train, np.array(x_test), dummy_y_test


def gen_clf_neupy(datapath):

    x_train, y_train, x_test, y_test = get_data(datapath)

    network = layers.join(
        layers.Input((48)),
        layers.Relu(20),
        layers.Softmax(36),
    )

    clf = algorithms.GradientDescent(
        network,
        step=0.1,
        shuffle_data = True,
        verbose = True,
        error = 'mse',
        # momentum=0.99
    )

    # x_train = np.array(x_train)
    # print x_train.shape

    clf.train(x_train, y_train, x_test, y_test, epochs=15)
    # plots.error_plot(clf)
    # print clf
    ypredicted = clf.predict(x_test)
    for i in range(0,len(y_test)):
        if np.argmax(y_test[i]) == np.argmax(ypredicted[i]):
            index = np.argmax(y_test[i])
            print "the character is: " + str(dict[str(index)])
    return clf



def demo(clf):
    # dict = json.load(open('./ftrNlabels/dict.txt'))
    while True:
        name = raw_input("enter image name: ")
        try:
            img = cv2.imread(name,0)
            # binarization takes list
            bitmap = prePro.segment_binarization([img])
            ftrs = ftExt.ftExt_edges(bitmap[0])
            print ftrs
            predicted = clf.predict(ftrs)
            print predicted
            print predicted.shape
            max = np.max(predicted[0])
            print max
            index = np.argmax(predicted)
            print "index: " + str(index)
            print "the character is: " + str(dict[str(index)])
        except:
            continue



if __name__ == '__main__':
    clf = gen_clf_neupy('ftrNlabels')
    demo(clf)