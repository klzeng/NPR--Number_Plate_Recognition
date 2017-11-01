# coding: utf-8
import cv2
import preProcess as prePro
import featureExt as ftExt
import os

def get_character_images():
    return





if __name__ == '__main__':
    plate = prePro.plate_localization('NP_image6.jpg', showProcess=True)
    characters = prePro.plate_sementation(plate)

    for each in characters:
        feature = ftExt.ftExt_edges(each)
        print feature
        print '\n'


    # try:
    #     os.mkdir("./trainingData")
    # except:
    #     pass
    # os.chdir("./trainingData")
    # i = 91
    # for each in characters:
    #     # cv2.imshow("char", each)
    #     # cv2.waitKey(0)
    #     name = str(i) + ".jpg"
    #     cv2.imwrite(name, each)
    #     i+=1

    # cv2.destroyAllWindows()