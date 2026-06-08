#!/usr/bin/env python
 
import cv2
import numpy as np
import argparse
import os
import glob

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision RAW file Recorder sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', '--mode', default="calib", help="Specify whether to calibrate or recitfy images")
    parser.add_argument(
        '-f', '--file', default="./calibration.yml", help="Path to where to store the calibration matrix")
    parser.add_argument(
        '-p', '--preview', type=bool, default=False, help="Whether to show image previews when calibrating")
    args = parser.parse_args()
    return args

def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
    print("Saved coefficients to file")

def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    print("Loaded coefficients to from")
    return [camera_matrix, dist_matrix]

def calibrate(args):
    # Defining the dimensions of checkerboard
    CHECKERBOARD = (4,6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []
    
    
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    
    # Extracting path of individual image stored in a given directory
    images = glob.glob('./*.png')
    # print(images)
    print("Finding corners")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        if args.preview:
            cv2.imshow('img',img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # h,w = img.shape[:2]
    
    """
    Performing camera calibration by
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the
    detected corners (imgpoints)
    """
    print("Calibrating Camera")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    return [ret, mtx, dist, rvecs, tvecs]

def undistort(matrix, dist):
    images = glob.glob('./*.png')

    for fname in images:
        img = cv2.imread(fname)

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w,h), 1, (w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imshow('Raw', img)
        cv2.imshow('Undistorted', dst)
        cv2.waitKey()

if __name__ == "__main__":
    args = parse_args()

    if args.mode == "calib":
        ret, mtx, dist, rvecs, tvecs = calibrate(args)
        save_coefficients(mtx, dist, args.file)
    elif args.mode == "dist":
        mtx, dist = load_coefficients(args.file)
        undistort(mtx, dist)