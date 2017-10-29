# coding: utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc

def plot_ProjPlot(projArr, peak, leftbound, rightbound):
    plt.plot(projArr)
    plt.axvline(x=peak, color='g')
    plt.axvline(x=leftbound, color='r')
    plt.axvline(x=rightbound, color='r')

"""
# So we used double phase method to locate the number plate, but in our way.
# - we compute convolution product of the source image then project rather then the reverse order
# - we use the "vertical edge filter" kernel matrix only, instead of horizontal/vertical rank filter
# - and the way we calculate the horizontal coordinate is made up by myself.(of course not target on specific case)
# - its not stable(very)
# - if we could figure out the the rank filter to do the convolution, we could easily follow the paper.
"""

def plate_localization(sourceImg):
    # read grey scale image
    img = cv2.imread(sourceImg, 0)
    # img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)

    # kernel matrice for convoution
    kernelX = np.array([-1,-1,-1,0,0,0,1,1,1]).reshape(3,3)
    kernelY = np.array([-1,0,1,-1,0,1,-1,0,1]).reshape(3,3)

    # compute the convolution using "vertical edge filter"
    imgConV = cv2.filter2D(img,-1,kernelY)

    # project to y-axis(sum of each row) and find the miximum
    yproject = imgConV.sum(axis = 1)
    yPeak = np.max(yproject)
    yFoot = 0.55*yPeak   # threshold 0.55

    # caluculat the vertical coordinate of the band
    yPeakIndex = np.argmax(yproject)
    ystart = 0
    yend = 0
    for each in range(0,yPeakIndex):
        if(yproject[each] <= yFoot):
            ystart = each

    for each in range(yPeakIndex,len(yproject)):
        if(yproject[each] <= yFoot):
            yend = each
            break

    plot_ProjPlot(yproject,yPeak,ystart,yend)
    # plt.show()

    # crop the band out
    img_band = img[ystart:yend,0:img.shape[1]]
    # cv2.imwrite('NP_image6_band.png',img_band)


    # do horizontal convolution on band
    bandConH = cv2.filter2D(img_band,-1,kernelY)

    # # x-axis project of band convolution(sum of each column)
    bandProjX = np.sum(bandConH, axis=0)
    xPeak = np.max(bandProjX)
    xPeakIndex = np.argmax(bandProjX)

    # caluculat the vertical coordinate of the band
    '''
    here we observe variance is very high in the region of the number plate(intensive peaks)
    - so , still, we set up the threshold, but find the minimum x < xPeakIndex, while bandProjX[x] >= threshold as 'left' boundary
    - and we find the maximum x > xPeakIndex, while bandProjX[x] >= threshold as 'right' boundary
    - indeed, its not stable
    '''
    xFoot = xPeak*0.55
    xstart = 0
    xend = 0
    for each in range(0,xPeakIndex):
        if(bandProjX[each] >= xFoot):
            xstart = each
            break

    for each in range(xPeakIndex,len(bandProjX)):
        if(bandProjX[each] >= xFoot):
            xend = each

    plot_ProjPlot(bandProjX,xPeakIndex,xstart,xend)
    # plt.show()

    # here we got the plate!
    img_crop = img_band[0:yend-ystart, xstart:xend]
    # cv2.imwrite("grey12.jpg",img_crop)
    # print img_crop.shape


    cv2.imshow("source", img)
    cv2.waitKey(0)

    cv2.imshow("img_band", img_band)
    cv2.waitKey(0)

    cv2.imshow("img_crop",img_crop)
    cv2.waitKey(0)

    return img_crop

def plate_sementation(plate):
    print plate.shape
    # for row in range(0,plate.shape[0]):
    #     for column in range(0,plate.shape[1]):
    #         if plate[row][column] >= 100:
    #             plate[row][column] = 255
    xProjction = np.sum(plate,axis=0)
    Vm = np.max(xProjction)
    Va = np.mean(xProjction)
    Vb = 2*Va - Vm
    Xm = np.argmax(xProjction)
    dividePoint = []
    Xl = 0
    Xr = 0
    while xProjction[Xm] >= 0.86*Vm:
        for x in range(Xm,0,-1):
            if xProjction[x] <= 0.7*xProjction[Xm]:
                Xl = x
                break
        for x in range(Xm,len(xProjction)):
            if xProjction[x] <= 0.8*xProjction[Xm]:
                Xr = x
                break
        for x in range(Xl, Xr):
            xProjction[x] =0
        dividePoint.append(Xm)
        Xm = np.argmax(xProjction)
    dividePoint.sort()
    characters = []
    Xr = 0
    for x in dividePoint:
        Xl = Xr
        Xr = x
        characters.append(plate[0:plate.shape[0], Xl:Xr])
    characters.append(plate[0:plate.shape[0],dividePoint[-1]:plate.shape[1]])
    return characters


if __name__ == '__main__':
    plate = plate_localization('NP_image12.jpg')
    characters = plate_sementation(plate)
    for each in characters:
        cv2.imshow("char", each)
        cv2.waitKey(0)
