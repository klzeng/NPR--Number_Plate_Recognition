# coding: utf-8
import cv2
from neupy import algorithms, layers,plots
import pandas
import os
import pre_process as prePro
import feature_ext as ftExt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import recall_score
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import np_utils


dict={'24': '4', '25': 'X', '26': '3', '27': 'E', '20': 'C', '21': 'D', '22': 'V',
      '23': 'Q', '28': 'B', '29': 'K', '1': 'U', '0': 'R', '3': '0', '2': '9', '5': 'I',
      '4': '7', '7': 'G', '6': 'N', '9': 'Z', '8': '6', '11': '8', '10': '1', '13': 'S', '12': 'T',
      '15': 'F', '14': 'A', '17': 'H', '16': 'O', '19': 'J', '18': 'M', '31': '2', '30': 'L', '35': 'W', '34': 'P', '33': '5', '32': 'Y'}

def get_data(datapath, hw=False):
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
                if not hw:
                    for key, value in dict.iteritems():
                        if value == letter:
                            label = int(key)
                else:
                    label = int(file.split('.')[0])
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



def gen_clf_KNN(datapath, hw=False, KNC=False):
    x_train, y_train, x_test, y_test, y_train_label, y_test_label = get_data(datapath, hw)
    print "x_train: " + str(x_train.shape)
    y_train_label.reshape(-1,1)
    print "y_train:" + str(y_train_label.shape)
    # print x_train
    # print y_train_label
    knn = cv2.ml.KNearest_create()
    knn.train(x_train,cv2.ml.ROW_SAMPLE, y_train_label)
    ret, result, neighbours, dist = knn.findNearest(x_test, k=5)
    matches = result== y_test_label.reshape(-1,1)
    correct = np.count_nonzero(matches)
    print correct
    print result.size, y_test_label.size
    accuracy = correct *100/result.size
    print "accuracy for KNN character recognition:"  + str(accuracy) + "%"

    estimator = KNeighborsClassifier()
    # scoring = ['precision_macro', 'recall_macro']
    scores = cross_validate(estimator, x_train, y_train_label, cv=5)
    # estimator.fit(x_train, y_train_label)
    # preditect = estimator.predict(x_test)
    # correct = np.count_nonzero(preditect)
    # print correct
    # print result.size, y_test_label.size
    # accuracy = correct *100/result.size
    # print "accuracy for KNC character recognition:"  + str(accuracy) + "%"
    print "train score: " + str(scores['train_score'])
    print "test score: " +  str(scores['test_score'])
    if KNC:
        return estimator
    else:
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

def demo_KNN(clf, hw=False):
    while True:
        name = raw_input("enter image name: ")
        try:
            img = cv2.imread(name,0)
            # binarization takes list
            if hw:
                (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            bitmap = prePro.segment_binarization([img])
            ftrs = ftExt.ftExt_edges(bitmap[0]).reshape(1,48)
            print "feature extracted:"
            print ftrs.reshape(6,8)
            ret, result, neighbours, dist = clf.findNearest(ftrs.astype(np.float32), k=5)
            if hw:
                print "result: "+ str(result[0][0])
            else:
                print "the character is: " + str(dict[str((int)(result[0][0]))])
        except:
            continue


def test_KNC(clf, hw = False):
    while True:
        name = raw_input("enter image name: ")
        try:
            img = cv2.imread(name,0)
            # binarization takes list
            if hw:
                (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.imshow("test", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            bitmap = prePro.segment_binarization([img])
            ftrs = ftExt.ftExt_edges(bitmap[0]).reshape(1,48).astype(np.float32)
            print "feature extracted:"
            print ftrs.reshape(6,8)
            # ret, result, neighbours, dist = clf.findNearest(ftrs.astype(np.float32), k=5)
            result = clf.predict(ftrs)
            # print result[0]
            if hw:
                print "result: "+ str(result[0])
            else:
                print "the character is: " + str(dict[str((int)(result[0]))])
        except:
            continue


def demo_NPR(clf, showProcess=False, KNC=False):
    while True:
        name = raw_input("enter image name: ")
        num_plate = ""
        try:
            plate = prePro.plate_localization(name, showProcess)
            characters, character_bitmaps = prePro.plate_sementation(plate, showProcess)
            for char in character_bitmaps:
                # binarization takes list
                ftrs = ftExt.ftExt_edges(char).reshape(1,48).astype(np.float32)
                # print ftrs
                if KNC:
                    result = clf.predict(ftrs)
                    num_plate = num_plate + str(dict[str((int)(result[0]))]) + " "
                else:
                    ret, result, neighbours, dist = clf.findNearest(ftrs, k=5)
                    num_plate = num_plate + str(dict[str((int)(result[0]))]) + " "
                # print result
            print "the plate is: " + num_plate
            cv2.destroyAllWindows()
        except:
            continue


if __name__ == '__main__':
    # clf = gen_clf_neupy('ftrNlabels')
    clf = gen_clf_KNN('ftrNlabels', hw=False)
    # demo_KNN(clf, hw=True)
    # test_KNC(clf, hw=True)
    demo_NPR(clf,showProcess=False, KNC=False)