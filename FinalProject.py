import os
import cv2 as cv
import numpy as np


def main():

    tempPathTrue = 'true-flowers'
    trueTemplates = loadImages(tempPathTrue)

    tempPathFalse = 'false-flowers'
    falseTemplates = loadImages(tempPathFalse)

    main_image = cv.imread('test/test4.jpg')

    main_image = cv.resize(main_image,(400,200))
    img_gray = deleteBackground(main_image)
    img_gray = edgeDetect(img_gray)
    img_gray = cv.cvtColor(img_gray, cv.COLOR_BGR2GRAY)

    #showAllArrayImages(falseTemplates)  # testing

    thresholdTrue = 0.5
    trueRectangles = checkIfExist(img_gray, trueTemplates,thresholdTrue)

    thresholdFalse = 0.6
    falseRectangles = checkIfExist(img_gray, falseTemplates, thresholdFalse)

    rectangleResult = deleteFalse(trueRectangles,falseRectangles)
    drawRectangle(rectangleResult,main_image)
    finalResult = imageProcess(main_image,600,400)


    cv.imshow('main_image result',finalResult)
    cv.waitKey(0)


def deleteFalse(trueLoc,falseLoc):
    for (xf, yf, wf, hf) in falseLoc:
        for i in trueLoc:
            (xt, yt, wt, ht) = i
            if (xf>=xt and yf>=yt):
                if(xf+wf<=xt+wt and yf+hf<=yt+ht):
                    trueLoc = np.delete(trueLoc,i)
    return trueLoc


def checkIfExist(img_gray, templates, threshold):

    totalRectangles = []
    for template in templates:
        width,height = template.shape[::-1]
        while template.shape[1]>img_gray.shape[1] or template.shape[0]>img_gray.shape[0]:
            width = int(template.shape[1] * 90 / 100)
            height = int(template.shape[0] * 90 / 100)
            template = cv.resize(template, (width, height))
        resRectangles = []
        while template.shape[1]>30:
            while template.shape[0]>30:
                wi, hi = template.shape[::-1]
                res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                loc = list(zip(*loc[::-1]))
                if loc:
                    r = allLocations(template,loc)
                    r = np.array(r).tolist()
                    resRectangles = resRectangles+r
                hi = int(template.shape[0] * 90 / 100)
                template = cv.resize(template, (wi,hi))
            # end while shape[0]
            hi = height
            wi = int(template.shape[1] * 90 / 100)
            template = cv.resize(template, (wi, hi))
        # end while shape[1]
        resRectangles, wi = cv.groupRectangles(resRectangles, groupThreshold=3, eps=0.7)
        resRectangles = np.array(resRectangles).tolist()
        totalRectangles = totalRectangles + resRectangles
    # end for loop on templates
    totalRectangles, wi = cv.groupRectangles(totalRectangles, groupThreshold=3, eps=0.7)
    totalRectangles = np.array(totalRectangles).tolist()

    return totalRectangles

def allLocations(template, locations):  # group location from 'locations' to 'rectangles'
    rectangles = []
    w = template.shape[0]
    h = template.shape[1]
    for location in locations:
        rect = [int(location[0]), int(location[1]), w, h]
        rectangles.append(rect)

    rectangles, wi = cv.groupRectangles(rectangles, groupThreshold=3, eps=0.7)
    return rectangles

def drawRectangle(rectangles, main_image):  # draw from the rectangles (array of location) on the image
    if len(rectangles):
        line_color = (0, 0, 255)
        line_type = cv.LINE_4
        thickness = 1
        for (x, y, w, h) in rectangles:
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            cv.rectangle(main_image, top_left, bottom_right, line_color, line_type, thickness)

def loadImages(path):
    images = []
    for filename in os.listdir(path):
        img = cv.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
    templates = []
    for i in images:
        k = deleteBackground(i)
        j = edgeDetect(k)
        j = cv.cvtColor(j, cv.COLOR_BGR2GRAY)
        templates.append(j)
    return templates

def showAllArrayImages(ImgArr):  # testing
    for i in ImgArr:
        print(j)
        cv.imshow('img array', i)
        cv.waitKey(0)


def edgeDetect(gray):
    grad_x = cv.Sobel(gray, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Scharr(gray,cv.CV_16S,0,1)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    result = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return result

def deleteBackground(srcImg):
    height, width = srcImg.shape[:2]
    mask = np.zeros(srcImg.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (10, 10, width - 30, height - 30)
    cv.grabCut(srcImg, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    resImg = srcImg * mask[:, :, np.newaxis]
    background = srcImg - resImg
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    result = background + resImg
    return result

def imageProcess(src,w,h):  # gauss and laplacian process, resize the image
    src = cv.resize(src,(w,h))
    src = cv.GaussianBlur(src, (3, 3), 0)
    sharpening_filter = np.array([[-1,-1,-1],
                                  [-1,9,-1],
                                  [-1,-1,-1]])
    afterFilterImage = cv.filter2D(src,-1,sharpening_filter)
    return afterFilterImage


if __name__ == "__main__":
    main()
