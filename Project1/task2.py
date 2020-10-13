###############
##Design the function "calibrate" to  return 
# (1) intrinsic_params: should be a list with four elements: [f_x, f_y, o_x, o_y], where f_x and f_y is focal length, o_x and o_y is offset;
# (2) is_constant: should be bool data type. False if the intrinsic parameters differed from world coordinates. 
#                                            True if the intrinsic parameters are invariable.
###############
import numpy as np
from cv2 import imread, cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

def calibrate(imgname):
        
    #Setting the stop criteria for the corner search with accuracy of 0.001
    criteria = (TERM_CRITERIA_EPS + TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.array([[4,0,4],[4,0,3],[4,0,2],[4,0,1],[3,0,4],[3,0,3],[3,0,2],[3,0,1],
                           [2,0,4],[2,0,3],[2,0,2],[2,0,1],[1,0,4],[1,0,3],[1,0,2],[1,0,1],
                           [0,1,4],[0,1,3],[0,1,2],[0,1,1],[0,2,4],[0,2,3],[0,2,2],[0,2,1],
                           [0,3,4],[0,3,3],[0,3,2],[0,3,1],[0,4,4],[0,4,3],[0,4,2],[0,4,1]],np.float32)
    
    #multiplying edge size to all points which is the distance b/n points
    objp *= 5 #5 mm is the size of an edge of check board
        
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    img=imread(imgname)
  
    gray=cvtColor(img,COLOR_BGR2GRAY)
    
    #find chessboard corners 
    found,corners=findChessboardCorners(gray,(4,9),None)
    found,corners=findChessboardCorners(gray,(4,9),None)
    
    if found==True:                
        #deleting the corners which are on the z axis to retain just 32 points
        corners  = np.delete(corners,tuple(range(16,20)),axis=0)
        corners2 = cornerSubPix(gray,corners,(4,4),(-1,-1),criteria)
        corners2 = cornerSubPix(gray,corners,(4,4),(-1,-1),criteria)
        
        imgpoints.append(corners2) #image points
        objpoints.append(objp)     #world points
        
        img = cv2.drawChessboardCorners(img, (4,4), corners2[0:15,:,:],found)
        img = cv2.drawChessboardCorners(img, (4,4), corners2[16:31,:,:],found)
        #cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
        #cv2.imshow('output',img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    #Now we have 32 image points and world coordinates 
    #to find coefficients solution of the equation Mx=0 we have linear set of equations as matrix 
    #lets apply SVD- single decompostion to get the coefficients of projection matrix.
    
    worldpoints=objpoints[0]
    imagepoints=imgpoints[0]
    imagepoints = imagepoints.reshape((imagepoints.shape[0]*imagepoints.shape[1]), imagepoints.shape[2])
    
    wrow,wcol=worldpoints.shape
    irow,icol=imagepoints.shape
     
    n=wrow
    PM=np.zeros((2*n,12))
    
    for i in range(n):
        
        X,Y,Z=worldpoints[i]
        x,y=imagepoints[i]
        
        #Set of linear equations
        eqn1 = np.array([  X,  Y,  Z,  1,  0,  0,  0,  0, -(X*x), -(Y*x), -(Z*x), -(x)])
        eqn2 = np.array([  0,  0,  0,  0,  X,  Y,  Z,  1, -(X*y), -(Y*y), -(Z*y), -(y)])
        PM[2*i] = eqn1
        PM[(2*i) + 1] = eqn2
    
    U, S, VT = np.linalg.svd(PM)
    
    xroots=VT[-1]
    lambda1=np.sqrt(1/np.sum(xroots[8:11]**2))
    mmat=lambda1*xroots    
    mmat=mmat.reshape(3,4)
    
    ox=np.sum(np.multiply(mmat[0,0:3],mmat[2,0:3]))
    oy=np.sum(np.multiply(mmat[1,0:3],mmat[2,0:3]))
    
    fx=np.sqrt(np.sum(mmat[0,0:3]**2)-ox**2)
    fy=np.sqrt(np.sum(mmat[1,0:3]**2)-oy**2)
      
    intrmat=[fx,fy,ox,oy]
    
    return intrmat,True
    

if __name__ == "__main__":
    intrinsic_params, is_constant = calibrate('checkboard.png')