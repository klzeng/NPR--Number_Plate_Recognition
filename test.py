import numpy as np
import cv2
import struct


def test():
    trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
    # Labels each one either Red or Blue with numbers 0 and 1
    responses = np.random.randint(0,2,(25,1)).astype(np.float32)
    print "train data: "  + str(trainData.shape)
    print "response" + str(responses.shape)
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses.reshape(-1,1))
    newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
    print newcomer
    ret, results, neighbours ,dist = knn.findNearest(newcomer,3)
    print results

if __name__ == '__main__':
    test()