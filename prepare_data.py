# coding: utf-8
import cv2
import preProcess as prePro
import featureExt as ftExt
import os
import numpy as np
import pandas as pd

def rename_train_char(rootpath):
    for subdir, dirs, files in os.walk(rootpath):
        for dir in dirs:
            print '\n\n' + dir
            path = rootpath + '/' + str(dir)
            for s, d, chars in os.walk(path):
                j = 30
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

def conver2BW(rootpath, writePath):
    for subdir, dirs, files in os.walk(rootpath):
        for dir in dirs:
            print '\n\n' + dir
            path = rootpath + '/' + str(dir)
            to = writePath + '/' + dir
            for s, d, chars in os.walk(path):
                j = 30
                for char in chars:
                    if char.endswith('jpg'):
                        oldname = path+'/'+ char
                        im_gray = cv2.imread(oldname,0)
                        (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                        os.chdir(writePath+ '/' + dir)
                        name =  dir + '_' + str(j) + '.jpg'
                        print name
                        cv2.imwrite(name, im_bw)
                        j = j + 1
                        os.chdir('../..')

def gen_ftrNlabels(rootpath, ftr_save_path):
    curIndex =0
    char2index_dict = {}
    for subdir, dirs, files in os.walk(rootpath):
        # dir is the name of the diretory contains the character, which is the character itself
        for dir in dirs:
            datas = []
            char2index_dict[str(curIndex)] = str(dir)
            print '\n\n' + dir + '  ' + str(curIndex)
            path = rootpath + '/' + str(dir)
            for s, d, chars in os.walk(path):
                for char in chars:
                    if char.endswith('jpg'):
                        # make it bitmap first
                        name = path+'/'+ char
                        print name
                        img = cv2.imread(name,0)
                        # binarization takes list
                        bitmap = prePro.segment_binarization([img])
                        ftrs = ftExt.ftExt_edges(bitmap[0])
                        ftrsNLabel = np.append(ftrs, curIndex)
                        datas.append(ftrsNLabel)
                curIndex = curIndex+1
            df = pd.DataFrame(data=datas)
            df.to_csv(ftr_save_path+"/"+dir+".csv", header=False)
    with open(ftr_save_path+"/dict.txt","wb") as out:
            out.write(str(char2index_dict))



# 472 473
if __name__ == '__main__':
    # extract_train_char('day_color(large sample)/U')
    # extract_from_sigle_plate()
    # rename_train_char('train10X20')
    gen_ftrNlabels('trainingData','ftrNlabels')
    # conver2BW('train10X20', 'trainingData')








