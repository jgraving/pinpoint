"""
Copyright 2015-2017 Jacob M. Graving <jgraving@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import cv2
import pickle
import glob

# disable multithreading in OpenCV for main thread
# to avoid problems after parallelization
cv2.setNumThreads(0)


class CameraCalibration:

    def __init__(self, grid_shape=(9, 6)):

        """ Class for calculating calibration parameters from calibration images

            Parameters
            ----------
            grid_shape : tuple of int
                Shape of calibration grid (internal corners)

            Returns
            -------
            self : class
                CameraCalibration class instance.

        """

        self.grid_shape = grid_shape

    def calibrate(self, image_files, imshow=True, delay=500):

        """ Calculates calibration parameters from calibration images

            Parameters
            ----------
            image_files : str
                File path to images (e.g. "/path/to/files/*.jpg")
            grid_shape : tuple of int
                Size of calibration grid (internal corners)
            imshow : bool, (default = True)
                Show the calibration images
            delay : int, >=1 (default = 500)
                Delay in msecs between each image for imshow

            Returns
            -------
            params : dict
                Parameters for undistorting images.

        """

        image_files = sorted(glob.glob(image_files))

        # termination criteria
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            300,
            0.001
        )

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.grid_shape[1] * self.grid_shape[0], 3),
                        np.float32)
        objp[:, :2] = np.mgrid[0:self.grid_shape[0],
                               0:self.grid_shape[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        for filename in image_files:
            img = cv2.imread(filename)

            if isinstance(img, None):

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv2.findChessboardCorners(
                    gray,
                    self.grid_shape,
                    None,
                    flags=(cv2.CALIB_CB_ADAPTIVE_THRESH +
                           cv2.CALIB_CB_FILTER_QUADS +
                           cv2.CALIB_CB_FAST_CHECK +
                           cv2.CALIB_CB_NORMALIZE_IMAGE))

                # If found, add object points,
                # image points (after refining them)
                if ret:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray,
                                                corners,
                                                (11, 11),
                                                (-1, -1),
                                                criteria
                                                )
                    imgpoints.append(corners2)

                    if imshow:
                        img = cv2.drawChessboardCorners(img,
                                                        self.grid_shape,
                                                        corners2,
                                                        ret
                                                        )

                # Draw and display the corners
                if imshow:
                    cv2.imshow('img', img)
                    cv2.waitKey(delay)

        if imshow:
            cv2.destroyAllWindows()
            for i in range(5):
                cv2.waitKey(1)

        if len(objpoints) > 0 and len(imgpoints) > 0:
            calibration = cv2.calibrateCamera(objpoints,
                                              imgpoints,
                                              gray.shape[::-1],
                                              None, None)
            ret, mtx, dist, rvecs, tvecs = calibration
            params = {"ret": ret,
                      "mtx": mtx,
                      "dist": dist,
                      "rvecs": rvecs,
                      "tvecs": tvecs}

            total_error = 0
            for i in xrange(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i],
                                                  rvecs[i],
                                                  tvecs[i],
                                                  mtx,
                                                  dist)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
                error /= len(imgpoints2)
                total_error += error
            mean_error = total_error / len(objpoints)

            print("Calibration successful! Mean error: ", mean_error)

            self.params = params
            self.mean_error = mean_error
            self.ret = ret
            self.mtx = mtx
            self.dist = dist
            self.rvecs = rvecs
            self.tvecs = tvecs

        else:
            print("No calibration points found!")
            self.params = None

        return self

    def undistort(self, image, crop=True):

        """ Returns undistorted image using calibration parameters.

            Parameters
            ----------
            image : numpy_array
                Image to be undistorted
            params : dict
                Calibration parameters
            crop : bool
                Crop the image to the optimal region of interest

            Returns
            -------
            dst : numpy_array
                Undistorted image.

        """

        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (w, h), 1, (w, h))

        # undistort
        mapx, mapy = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        # crop the image
        if crop:
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

        return dst

    def save_calib(self, filename):
        """ Saves calibration parameters as '.pkl' file.

            Parameters
            ----------
            filename : str
                Path to save file, must be '.pkl' extension

            Returns
            -------
            saved : bool
                Saved successfully.
        """
        if type(self.params) != dict:
                raise TypeError("params must be 'dict'")

        output = open(filename, 'wb')

        pickle.dump(self.params, output, protocol=0)

        output.close()

        self.saved = True

        return self.saved

    def load_calib(self, filename):
        """ Loads calibration parameters from '.pkl' file.

            Parameters
            ----------
            filename : str
                Path to load file, must be '.pkl' extension

            Returns
            -------
            params : dict
                Parameters for undistorting images.

        """
        # read python dict back from the file

        pkl_file = open(filename, 'rb')

        self.params = pickle.load(pkl_file)

        pkl_file.close()

        self.ret = self.params["ret"]
        self.mtx = self.params["mtx"]
        self.dist = self.params["dist"]
        self.rvecs = self.params["rvecs"]
        self.tvecs = self.params["tvecs"]

        self.loaded = True
        return self.loaded
