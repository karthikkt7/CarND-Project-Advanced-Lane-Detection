import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

###Functions for transformation###
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def bgr2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def bgr2hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)	

def bgr_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def cameraCalib(objpoints, imgpoints, shape):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape,None,None)
    return mtx, dist, rvecs, tvecs

def UndistImg(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)
	
#Used from ./examples/example.ipynb
def FindChessCorners(Imgs):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    imgs = []

    # Step through the list and search for chessboard corners
    for img in Imgs:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            #cv2 --> BGR format
            img = bgr2rgb(cv2.drawChessboardCorners(img, (9,6), corners, True))
            imgs.append(img)
            
    return objpoints, imgpoints, imgs



