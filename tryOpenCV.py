# coding: utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('NP_image6.jpg',0)
print img.shape
kernelH = np.array([-1,-1,-1,0,0,0,1,1,1])
kernelV = np.array([-1,0,1,-1,0,1,-1,0,1])
imgConV = cv2.filter2D(img,-1,kernelV)

# sum of each row
# print "ori y proj\n"
# print img.sum(axis = 1)
print "convo y proj"
yproject = imgConV.sum(axis = 1)
yPeak = np.max(yproject)
yPeakIndex = np.argmax(yproject)
yFoot = 0.55*yPeak
print yPeak, yFoot

# caluculat the vertical coordinate of the band
ystart = 0
yend = 0
for each in range(0,yPeakIndex):
    if(yproject[each] <= yFoot):
        ystart = each

for each in range(yPeakIndex,len(yproject)):
    if(yproject[each] <= yFoot):
        yend = each
        break

# crop the band out
img_band = img[ystart:yend,0:img.shape[1]]

# do horizontal convolution on band(sum of each column)
bandConH = cv2.filter2D(img_band,-1,kernelH)

# x-axis project of band convolution
bandProjX = np.sum(bandConH, axis=0)
xPeak = np.max(bandProjX)
xPeakIndex = np.argmax(bandProjX)
print xPeakIndex
xFoot = xPeak*0.55
xstart = 0
xend = 0
for each in range(0,xPeakIndex):
    if(bandProjX[each] <= xFoot):
        xstart = each

for each in range(xPeakIndex,len(bandProjX)):
    if(bandProjX[each] <= xFoot):
        xend = each
        break
print "xstart: " + str(xstart)
print "xend: " + str(xend)
img_crop = img_band[0:yend-ystart, xstart:xend]
print img_crop.shape

cv2.imshow("img_crop",img_band)
cv2.waitKey(0)


# (thresh, img_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     hist =  cv2.calcHist([img], [0], None, [256], [0,256])
#     print hist
#     plt.hist(img.ravel(),256,[0,256])
#     plt.show()
# cv2.imshow("win1", img_bw)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


