# coding: utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('NP_image2.jpg',0)
print img.shape
kernel = np.array([0,0,0,0,1,1,0,0,0])
kernelV = np.array([-1,0,1,-1,0,1,-1,0,1])
imgConV = cv2.filter2D(img,-1,kernelV)

# sum of each row
# print "ori y proj\n"
yProject = img.sum(axis = 1)
# do convolution on it
yProjCon = np.convolve(yProject,kernel,"same")
yPeak = np.max(yProjCon)
yPeakIndex = np.argmax(yProjCon)
yFoot = 0.55*yPeak
# print yProjCon
# print yPeak, yFoot, yPeakIndex


# # caluculat the vertical coordinate of the band
ystart = 0
yend = 0
for each in range(0,yPeakIndex):
    if(yProjCon[each] <= yFoot):
        ystart = each

for each in range(yPeakIndex,len(yProjCon)):
    if(yProjCon[each] <= yFoot):
        yend = each
        break

plt.plot(yProjCon)
plt.axvline(x=yPeakIndex, color='g')
plt.axvline(x=ystart, color='r')
plt.axvline(x=yend, color='r')
plt.show()
# crop the band out
print ystart, yend
img_band = img[ystart:yend,0:img.shape[1]]
print img_band.shape
# cv2.imshow("6",img_band)
# cv2.waitKey(0)