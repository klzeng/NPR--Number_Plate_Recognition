# coding: utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt


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

def plate_localization(sourceImg, showProcess = False):
    # read grey scale image
    img = cv2.imread(sourceImg, 0)
    if img is None:
        print "failed to read image.\n"
        exit(1)
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

    # plot_ProjPlot(yproject,yPeak,ystart,yend)
    # if showProcess:
    #     plt.title("y project of ori image")
    #     plt.show()

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
    # if showProcess == True:
    #     plt.title("x project of band")
    #     plt.show()

    # here we got the plate!
    img_crop = img_band[0:yend-ystart, xstart:xend]
    # cv2.imwrite("grey12.jpg",img_crop)
    # print img_crop.shape

    if showProcess:
        cv2.imshow("source", img)
        cv2.waitKey(0)
        cv2.imshow("img_band", img_band)
        cv2.waitKey(0)
        cv2.imshow("img_crop",img_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_crop

def plate_sementation(plate, showProcess=False):
    # print plate.shape
    # for row in range(0,plate.shape[0]):
    #     for column in range(0,plate.shape[1]):
    #         if plate[row][column] >= 100:
    #             plate[row][column] = 255
    xProjction = np.sum(plate,axis=0)
    Vm = np.max(xProjction)
    Va = np.mean(xProjction)
    Vb = 2*Va - Vm
    Xm = np.argmax(xProjction)
    Xl = Xm
    Xr = Xm
    dividePoint = []
    x = 0
    # add an list of the index of already zerod region
    zeroedIndex = []
    while xProjction[Xm] >= 0.86*Vm:
        lowBound = Xm
        highBound = Xm
        if len(zeroedIndex)==0:
            lowBound = 0
            highBound = len(xProjction)
        else:
            zeroedIndex.sort()
            if(zeroedIndex[-1] < Xm):
                highBound = len(xProjction)
                lowBound = zeroedIndex[-1] + 1
            if(zeroedIndex[0] > Xm):
                lowBound =0
                highBound = zeroedIndex[0] - 1
            if highBound == Xm or lowBound == Xm:
                for each in zeroedIndex:
                    if each < Xm:
                        lowBound = each
                    if highBound == Xm and each > highBound:
                        highBound = each
        for x in range(Xm,lowBound,-1):
            if xProjction[x] <= 0.7*xProjction[Xm]:
                Xl = x
                break
        if x == lowBound+1:
            Xl = lowBound
        for x in range(Xm,highBound):
            if xProjction[x] <= 0.8*xProjction[Xm]:
                Xr = x
                break
        if x == highBound-1:
            Xr = highBound
        for x in range(Xl, Xr):
            xProjction[x] =0
        zeroedIndex.append(Xl)
        zeroedIndex.append(Xr)
        if Xl != lowBound and Xr != highBound:
            dividePoint.append(Xm)
        Xm = np.argmax(xProjction)
        # print "zeroedIndex: " + str(zeroedIndex) + '\n'
        # print "dividePoint: " + str(dividePoint) + '\n'
    dividePoint.sort()
    segments = []
    Xr = 0
    for x in dividePoint:
        Xl = Xr
        Xr = x
        segments.append(plate[0:plate.shape[0], Xl:Xr])
    segments.append(plate[0:plate.shape[0],dividePoint[-1]:plate.shape[1]])
    bitmaps = segment_binarization(segments)
    if showProcess:
        for each in bitmaps:
            # print each
            cv2.imshow("char", each)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    return segments, bitmaps

# return 0,1 bitmap, 0-black, 1-white
def segment_binarization(segments):
    segment_bitmaps = []
    for segment in segments:
        bitmap = np.zeros(segment.shape)
        for row in range(0,segment.shape[0]):
            for column in range(0,segment.shape[1]):
                if segment[row][column] <= 110:
                    segment[row][column] = 0
                else:
                    segment[row][column] = 255
                    bitmap[row][column] = 1
        segment_bitmaps.append(bitmap)
    return segment_bitmaps

# for digits dataset binarization
def segment_binarization_digits(segments):
    segment_bitmaps = []
    for segment in segments:
        bitmap = np.zeros(segment.shape)
        for row in range(0,segment.shape[0]):
            for column in range(0,segment.shape[1]):
                if segment[row][column] == 0:
                    bitmap[row][column] = 1
                else:
                    bitmap[row][column] = 0
        segment_bitmaps.append(bitmap)
    return segment_bitmaps


#./day_color(large sample)/HPIM0596.JPG
if __name__ == '__main__':
    plate = plate_localization('NP_image14.jpg', showProcess=False)
    characters = plate_sementation(plate)
    for each in characters:
        # print each
        cv2.imshow("char", each)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    # a = range(6)
    # print a
    # print a.index(9)
