###############
##1. Design the function "rectify" to  return
# fundamentalMat: should be 3x3 numpy array to indicate fundamental matrix of two image coordinates. 
# Please check your fundamental matrix using "checkFunMat". The mean error should be less than 5e-4 to get full point.
##2. Design the function "draw_epilines" to  return
# draw1: should be numpy array of size [imgH, imgW, 3], drawing the specific point and the epipolar line of it on the left image; 
# draw2: should be numpy array of size [imgH, imgW, 3], drawing the specific point and the epipolar line of it on the right image.
# See the example of epilines on the PDF.
###############
from cv2 import imread, xfeatures2d, FlannBasedMatcher, cvtColor, COLOR_RGB2BGR, line, circle, computeCorrespondEpilines
import numpy as np
from matplotlib import pyplot as plt

def rectify(pts1, pts2):
    #...



def draw_epilines(img1, img2, pt1, pt2, fmat):
    #...



def checkFunMat(pts1, pts2, fundMat):
    N = len(pts1)
    assert len(pts1)==len(pts2)
    errors = []
    for n in range(N):
        v1 = np.array([[pts1[n][0], pts1[n][1], 1]])#size(1,3)
        v2 = np.array([[pts2[n][0]], [pts2[n][1]], [1]])#size(3,1)
        error = np.abs((v1@fundMat@v2)[0][0])
        errors.append(error)
    error = sum(errors)/len(errors)
    return error
    
if __name__ == "__main__":
    img1 = imread('rect_left.jpeg') 
    img2 = imread('rect_right.jpeg')

    # find the keypoints and descriptors with SIFT
    sift = xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters for points match
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    dis_ratio = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.3*n.distance:
            good.append(m)
            dis_ratio.append(m.distance/n.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    min_idx = np.argmin(dis_ratio) 
    
    # calculate fundamental matrix and check error
    fundMat = rectify(pts1, pts2)
    error = checkFunMat(pts1, pts2, fundMat)
    print(error)
    
    # draw epipolar lines
    draw1, draw2 = draw_epilines(img1, img2, pts1[min_idx], pts2[min_idx], fundMat)
    
    # save images
    fig, ax = plt.subplots(1,2,dpi=200)
    ax=ax.flat
    ax[0].imshow(draw1)
    ax[1].imshow(draw2)
    fig.savefig('rect.png')