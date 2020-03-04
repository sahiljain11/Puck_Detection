import cv2
import numpy as np
import imutils
import argparse
import glob

objp = np.zeros((6 * 8, 3), np.float32)    #7x9 chessboard. can adjust later
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)      #(0,0,0), (1,0,0), ..., (6,5,0)
objp = objp * 2.5       #each square is 2.5 cm long
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []      #3d points
imgpoints = []      #2d points

images = glob.glob("chessboard\*.jpg")

for fname in images:
    img = cv2.imread(fname)
    #img = imutils.resize(img, width=600)  # increases FPS
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        #cv2.imshow("image", img)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
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
    fname = fname.split('\\')[1]
    cv2.imwrite('undistorted\\' + fname, dst)

    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))
