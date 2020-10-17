import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    src = cv.imread('project1/view1.jpg')
    #src = cv.imread('project1/view2.jpg')
    src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    if src is None:
        print ('Error opening image')
        print ('The image not exist in the folder')
        return -1

    filter = np.ones((5, 3), dtype=np.float) / 200
    r=my_imfilter(src,filter)

    plt.imshow(src, cmap='gray')
    plt.figure()
    plt.imshow(r, cmap='gray')
    plt.show()
    return 0

def my_imfilter(img, filter):
    img = img.astype(np.uint8)
    img_copy = np.zeros_like(img)
    img = zeroPadding(img)
    lapla_filter = np.array([[0,-1,0], [-1, 4, -1], [0,-1,0]])
    after_laplacian = filtering(img,img_copy,lapla_filter)
    after_filter = filtering(after_laplacian,img_copy,filter)
    return after_filter

def filtering(img_src,img_new,filter):
    filter_row = np.size(filter,0)
    filter_col = np.size(filter,1)
    for i in range(np.size(img_src, 0)-(filter_row-1)):
        for j in range(np.size(img_src, 1)-(filter_col - 1)):
            x = np.sum(filter * img_src[i: i + filter_row, j:j + filter_col])
            if x + img_src[i,j] > 255:
                img_new[i, j] = 255
            elif x + img_src[i,j] < 0:
                img_new[i, j] = 0
            else: img_new[i, j] = x + img_src[i,j]
    return img_new

def zeroPadding(img):
    padding = np.pad(img, ((1,1),(1,1)),'constant')
    return padding

if __name__ == "__main__":
    main()