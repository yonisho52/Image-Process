import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main(): #eye4.jpg + view2.jpg 0.5 , wall1 + wall5 0.7, face3.jpg + 111.jpg 0.6
    src1 = cv.imread('project2/eye4.jpg')
    src2 = cv.imread('project2/view2.jpg')
    if src1 is None or src2 is None:
        print ('Error opening image')
        print ('The image not exist in the folder')
        return -1
    dim = (600, 500)
    dimSource = (np.size(src1,1), np.size(src1,0))
    resized1 = cv.resize(src1, dim, interpolation=cv.INTER_AREA)
    resized2 = cv.resize(src2, dim, interpolation=cv.INTER_AREA)
    grey1 = cv.cvtColor(resized1, cv.COLOR_BGR2GRAY)
    grey2 = cv.cvtColor(resized2, cv.COLOR_BGR2GRAY)
    afterFilter1 = imageProcess(grey1)
    afterFilter2 = imageProcess(grey2)
    ratio = 0.6
    r=my_morphing(afterFilter1, afterFilter2, ratio)
    resized3 = cv.resize(r,dimSource, interpolation=cv.INTER_AREA)
    sharper = imageProcess(resized3)
    plt.imshow(sharper, cmap='gray'), plt.axis("off")
    plt.show()
    return 0

def my_morphing(s1,s2,ratio):
    #dst = cv.addWeighted(s1, ratio, s2, 1.0-ratio, 0.0)
    dst = np.zeros_like(s1)
    for i in range(np.size(dst, 0)):
        for j in range(np.size(dst, 1)):
            dst[i,j] = np.sum(s1[i,j]*ratio+s2[i,j]*(1-ratio) + 0.0)
    return dst

def imageProcess(src):  # gauss and laplacian process, resize the image
    src = cv.GaussianBlur(src, (3, 3), 0)
    srcFilter = cv.Laplacian(src, cv.CV_16S, 3)
    srcFilter1 = cv.convertScaleAbs(srcFilter)
    afterFilterImage = cv.add(src, srcFilter1)
    return afterFilterImage

if __name__ == "__main__":
    main()