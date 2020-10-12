###############
##Design the function "findRotMat" to  return 
# 1) rotMat1: a 2D numpy array which indicates the rotation matrix from xyz to XYZ 
# 2) rotMat2: a 2D numpy array which indicates the rotation matrix from XYZ to xyz 
###############

import numpy as np
import cv2

def findRotMat(alpha, beta, gamma):
    t1=np.pi*alpha/180
    t2=np.pi*beta/180
    t3=np.pi*gamma/180
    
    rotMat1=np.array([[np.cos(t1)*np.cos(t3)-np.sin(t1)*np.sin(t3)*np.cos(t2),-np.sin(t1)*np.cos(t3)-np.cos(t1)*np.cos(t2)*np.sin(t3),np.sin(t2)*np.sin(t3)],
                      [np.cos(t1)*np.sin(t3)+np.sin(t1)*np.cos(t2)*np.cos(t3),-np.sin(t1)*np.sin(t3)+np.cos(t1)*np.cos(t2)*np.cos(t3),-np.sin(t2)*np.cos(t3)],
                      [np.sin(t1)*np.sin(t2),np.cos(t1)*np.sin(t2),np.cos(t2)]])
    
    rotMat2=np.linalg.inv(rotmatptop1)
    
    return rotMat1,rotMat2


if __name__ == "__main__":
    alpha = 45
    beta = 30
    gamma = 50
    rotMat1, rotMat2 = findRotMat(alpha, beta, gamma)