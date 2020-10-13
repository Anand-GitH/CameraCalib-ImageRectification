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

    pts1=np.array(pts1)
    pts2=np.array(pts2)
        
    x=np.array(pts1[:,0])
    y=np.array(pts1[:,1])
    x1=np.array(pts2[:,0])
    y1=np.array(pts2[:,1])
    
    x=x.reshape(x.shape[0],1)
    y=y.reshape(y.shape[0],1)
    x1=x1.reshape(x1.shape[0],1)
    y1=y1.reshape(y1.shape[0],1)
    
    n=x.shape[0]
    M=np.empty((n,9))
    
    #Eight point algorithm - Linear combination of equations to calculate fundamental matrix AF=0
    for i in range(n):
        M[i]=np.array([x1[i]*x[i], x1[i]*y[i], x1[i], y1[i]*x[i], y1[i]*y[i], y1[i], x[i], y[i], 1])
    
    #AF=0 solve for F using SVD
    U, S, VT = np.linalg.svd(M)

    funmat=VT[-1]
    funmat=funmat.reshape(3,3).T
    
    return funmat

def draw_epilines(img1, img2, pt1, pt2, fmat):
    
    img1 = cvtColor(img1,COLOR_RGB2BGR)
    img2 = cvtColor(img2,COLOR_RGB2BGR)
    color=(0,255,0)
    
    #Find and mark the points and epilines on left image- Triangularization
    ll=computeCorrespondEpilines(np.array([list(pt2)]),1,fmat)
    ll = ll.reshape(3)
    lr, lc, ld = img1.shape 
    x, y = map(int, [0, -ll[2] / ll[1]]) 
    x1,y1 = map(int,[lc, -(ll[2] + ll[0] * lc) / ll[1]]) 
    img1 = line(img1,(x, y), (x1, y1), color, 2) 
    img1 = circle(img1,tuple(np.float32(pt1)), 10, color, -1) 
    
    
    #Find and mark the points and epilines on right image- Triangularization
    rl=computeCorrespondEpilines(np.array([list(pt1)]),2,fmat)
    rl=rl.reshape(3)
    rr, rc, rd = img2.shape 
    x, y = map(int, [0, -rl[2] / rl[1]]) 
    x1,y1 = map(int,[rc, -(rl[2] + rl[0] * rc) / rl[1]]) 
    img2 = line(img2,(x, y), (x1, y1), color, 2) 
    img2 = circle(img2,tuple(np.float32(pt2)), 10, color, -1) 

    return img1,img2

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