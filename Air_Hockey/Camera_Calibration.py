import cv2
import numpy as np
import imutils
import argparse
import glob

objp = np.zeros((6 * 8, 3), np.float32)    #7x9 chessboard. can adjust later
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)      #(0,0,0), (1,0,0), ..., (6,5,0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []
imgpoints = []

images = glob.glob("chessboard\*.jpg")

for fname in images:
    img = cv2.imread(fname)
    img = imutils.resize(img, width=1000)  # increases FPS
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    #cv2.imshow("image", img)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

for fname in images:
    img = cv2.imread(fname)
    img = imutils.resize(img, width=1000)  # increases FPS
    h, w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow("dst", dst)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
