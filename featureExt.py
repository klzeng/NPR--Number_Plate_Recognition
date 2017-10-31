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
- 26 letters and 10 digits, 36 potential candidates
- feature vector: 6*36
'''
def ftExt_edges(character):
