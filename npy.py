# coding: utf-8
import cv2
import pre_process as prePro
import feature_ext as ftExt
import os

def get_character_images():
    return


# 472 473
if __name__ == '__main__':

    i = 91
    rootpath = "./day_color(large sample)/"
    for subdir, dirs, files in os.walk(rootpath):
        for file in files:
            path = rootpath + file
            # print path
            try:
                plate = prePro.plate_localization(path, showProcess=False)
                characters, bitmaps = prePro.plate_sementation(plate)

                # for each in characters:
                #     feature = ftExt.ftExt_edges(each)
                #     print feature
                #     print '\n'

                try:
                    os.mkdir("./trainingData")
                except:
                    pass
                os.chdir("./trainingData")
                i = i+1
                for each in characters:
                    # print each
                    # cv2.imshow("char", each)
                    # cv2.waitKey(0)
                    name = str(i) + ".jpg"
                    cv2.imwrite(name, each)
                    i+=1
                os.chdir("..")
            except:
                print "failure for " + path
                continue










