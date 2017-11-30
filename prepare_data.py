# coding: utf-8
import cv2
import preProcess as prePro
import featureExt as ftExt
import os

def rename_train_char(rootpath):
    for subdir, dirs, files in os.walk(rootpath):
        for dir in dirs:
            print '\n\n' + dir
            path = rootpath + '/' + str(dir)
            for s, d, chars in os.walk(path):
                j =0
                for char in chars:
                    if char.endswith('jpg'):
                        oldname = path+'/'+ char
                        # print oldname
                        name = path + '/' + dir + '_' + str(j) + '.jpg'
                        # print name
                        os.rename(oldname, name)
                        j = j + 1


def extract_train_char(rootpath):
    i = 1100
    for subdir, dirs, files in os.walk(rootpath):
        for file in files:
            path = rootpath + '/' + file
            print path
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



def extract_from_sigle_plate():
    i=40
    plate = prePro.plate_localization('NP_image26.jpg', showProcess=False)
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

# 472 473
if __name__ == '__main__':
    # extract_train_char('day_color(large sample)/U')
    # extract_from_sigle_plate()
    rename_train_char('trainingData')








