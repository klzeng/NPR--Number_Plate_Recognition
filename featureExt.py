# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt

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
    character = patch_bitmap(character)
    # 6 regions totally
    x = np.zeros(48)
    for r in range(0,5):
        for i in range(0, character.shape[0]-1):
            for j in range(0, character.shape[1]-1):
                unit = np.array([character[i][j], character[i][j+1], character[i+1][j],character[i+1][j+1]]).reshape(2,2)
                c = match_edge(unit)
                if c == -1:
                    continue
                index = r*8 + c
                x[index] += 1
    return x

def patch_bitmap(bitmap):
    # print bitmap.shape
    expanded = np.ones((bitmap.shape[0]+1, bitmap.shape[1]+1), dtype=int)
    # print expanded.shape
    for i in range(1,expanded.shape[0]-1):
        for j in range(1, expanded.shape[1]-1):
            expanded[i][j] = bitmap[i][j]
    return expanded

# 14 edge type, grouped to 8 types
def match_edge(unit):
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
    index = -1
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


'''

'''