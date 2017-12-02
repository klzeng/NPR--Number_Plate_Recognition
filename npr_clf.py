# coding: utf-8
import cv2
from neupy import algorithms, layers,plots
import pandas
import os
import preProcess as prePro
import featureExt as ftExt
import numpy as np
from sklearn.preprocessing import LabelEncoder
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
    y_train_label = []
    y_test_label = []

    for subdir, dir, files in os.walk(datapath):
        for file in files:
            if file.endswith('csv'):
                df = pandas.read_csv(datapath+'/'+file, header=None)
                datas = df.values
                X = datas[:,1:49]
                Y = datas[:,-1]
                len = X.shape[0]
                testcount = (int)(0.9*len)
                letter = file.split('.')[0]
                label = -1
                for key, value in dict.iteritems():
                    if value == letter:
                        label = int(key)
                if label == -1:
                    raise
                print "for " + file + " " + str(testcount) + "/" + str(len)
                for i in range(testcount):
                    x_train.append(X[i].astype(np.float32))
                    y_train.append(Y[i])
                    y_train_label.append(np.array(label).astype(np.float32))
                for i in range(testcount,len):
                    x_test.append(X[i].astype(np.float32))
                    y_test.append(Y[i])
                    y_test_label.append(np.array(label).astype(np.float32))

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train_encoded = encoder.transform(y_train)
    encoder.fit(y_test)
    y_test_encoded = encoder.transform(y_test)
    dummy_y_train = np_utils.to_categorical(y_train_encoded)
    dummy_y_test = np_utils.to_categorical(y_test_encoded)

    return np.array(x_train), dummy_y_train, np.array(x_test), dummy_y_test,\
           np.array(y_train_label), np.array(y_test_label)


def gen_clf_neupy(datapath):

    x_train, y_train, x_test, y_test , y_train_label, y_test_label= get_data(datapath)

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



def gen_clf_KNN(datapath):
    x_train, y_train, x_test, y_test, y_train_label, y_test_label = get_data(datapath)
    print "x_train: " + str(x_train.shape)
    y_train_label.reshape(-1,1)
    print "y_train:" + str(y_train_label.shape)
    # print x_train
    # print y_train_label
    knn = cv2.ml.KNearest_create()
    knn.train(x_train,cv2.ml.ROW_SAMPLE, y_train_label)
    ret, result, neighbours, dist = knn.findNearest(x_test, k=5)
    matches = result== y_test_label
    correct = np.count_nonzero(matches)
    # print correct
    # accuracy = correct*100.0/result.size
    # print "accuracy:"  + str(accuracy)
    return knn


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

def demo_KNN(clf):
    while True:
        name = raw_input("enter image name: ")
        try:
            img = cv2.imread(name,0)
            # binarization takes list
            bitmap = prePro.segment_binarization([img])
            ftrs = ftExt.ftExt_edges(bitmap[0]).reshape(1,48)
            print ftrs
            ret, result, neighbours, dist = clf.findNearest(ftrs.astype(np.float32), k=5)
            print result
            print "the character is: " + str(dict[str((int)(result[0][0]))])
        except:
            continue


def demo_NPR(clf, showProcess=False):
    while True:
        name = raw_input("enter image name: ")
        num_plate = ""
        try:
            plate = prePro.plate_localization(name, showProcess)
            characters, character_bitmaps = prePro.plate_sementation(plate, showProcess)
            for char in character_bitmaps:
                # binarization takes list
                ftrs = ftExt.ftExt_edges(char).reshape(1,48)
                # print ftrs
                ret, result, neighbours, dist = clf.findNearest(ftrs.astype(np.float32), k=5)
                # print result
                num_plate = num_plate + str(dict[str((int)(result[0][0]))]) + " "
            print "the plate is: " + num_plate
        except:
            continue





if __name__ == '__main__':
    # clf = gen_clf_neupy('ftrNlabels')
    clf = gen_clf_KNN('ftrNlabels')
    # demo_KNN(clf)
    demo_NPR(clf,showProcess=True)