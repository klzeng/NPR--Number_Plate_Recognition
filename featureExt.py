# coding: utf-8

import numpy as np
import cv2
import preProcess as prePro

'''
We will have two different features:
 - detection of character edges
 - structural feature
'''


'''
Character edges features:
- divide to 6 rectangular regions
- 8 edge types(of 14 differnt bitmap )

- feature vector: 6*8
-
- input character should be 0,1 bitmap, 0 for black and 1 for white
- return is the featuer vector for input image
'''
def ftExt_edges(character):
    bitmap = patch_bitmap(character)
    # 6 regions totally
    ftr = np.zeros(48)
    regionWidth = (int)(bitmap.shape[1]/2)
    regionHeight = (int)(bitmap.shape[0]/3)
    for x in range(0,3):
        for y in range(0,2):
            for row in range(x*regionHeight, (x+1)*regionHeight-1):
                for column in range(y*regionWidth, (y+1)*regionWidth-1):
                    unit = np.array([bitmap[row][column], bitmap[row][column+1], bitmap[row+1][column],bitmap[row+1][column+1]]).reshape(2,2)
                    # print unit
                    c = match_edge(unit)
                    if c == -1:
                        continue
                    index = (x*2 + y)*8 + c - 1
                    ftr[index] += 1
    return ftr

def patch_bitmap(bitmap):
    # print bitmap.shape
    expanded = np.ones((bitmap.shape[0]+2, bitmap.shape[1]+2), dtype=int)
    # print expanded.shape
    for i in range(1,expanded.shape[0]-2):
        for j in range(1, expanded.shape[1]-2):
            expanded[i][j] = bitmap[i][j]
    return expanded




#  vertical edge
h0 = np.array([1,0,1,0]).reshape(2,2)
h1 = np.array([0,1,0,1]).reshape(2,2)
# horizontal edge
h2 = np.array([1,1,0,0]).reshape(2,2)
h3 = np.array([0,0,1,1]).reshape(2,2)
# "/" diagonal edge
h4 = np.array([1,0,0,1]).reshape(2,2)
h5 = np.array([1,0,0,0]).reshape(2,2)
h6 = np.array([0,0,0,1]).reshape(2,2)
# "\"
h7 = np.array([0,1,1,0]).reshape(2,2)
h8 = np.array([0,1,0,0]).reshape(2,2)
h9 = np.array([0,0,1,0]).reshape(2,2)
# bottom right corner
h10 = np.array([0,1,1,1]).reshape(2,2)
# bottom left corner
h11 = np.array([1,0,1,1]).reshape(2,2)
# top right corner
h12 = np.array([1,1,0,1]).reshape(2,2)
# top left corner
h13 = np.array([1,1,1,0]).reshape(2,2)

hs = [h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13]

# 14 edge type, grouped to 8 types
def match_edge(unit):
    index = 0
    for each in hs:
        if np.array_equal(each, unit):
            # index =  hs.index(each)
            break
        index+=1
    if index in range(0,2):
        return 1
    if index in range(2,4):
        return 2
    if index in range(4,7):
        return 3
    if index in range(7,10):
        return 4
    if index == 10:
        return 5
    if index == 11:
        return 6
    if index == 12:
        return 7
    if index == 13:
        return 8
    return -1



def debug(name):
    img = cv2.imread(name,0)
    # binarization takes list
    bitmap = prePro.segment_binarization([img])
    ftrs = ftExt_edges(bitmap[0])
    print ftrs.reshape(6,8)

if __name__ == '__main__':
    debug('I_10.jpg')