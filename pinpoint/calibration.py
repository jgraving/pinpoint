import numpy as np
import cv2
import pickle

def get_calib_video(calib_video, grid_size = (6,10), imshow = True, delay = 50):

	""" Returns calibration parameters from calibration images

		Parameters
		----------
		calib_video : str
			File path to video (e.g. "/path/to/video/video.mov").
		imshow : bool
			Show the calibration video
		delay : int, >=1
			Delay in msecs between each frame for imshow
			
		Returns
		-------
		calib_params : dict
			Parameters for undistorting images.

	"""
	cap = cv2.VideoCapture(calib_video)
	nframe = 0
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:grid_size[1],0:grid_size[0]].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	while cap.isOpened():

		img_ret, img = cap.read()

		nframe = nframe + 1

		if img_ret == True:

			frame = img.copy()

			blur = cv2.GaussianBlur(img,(21,21),0)
			img = cv2.resize(img, (0,0), None, fx = 0.25, fy = 0.25)

			img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Find the chess board corners
			corner_ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
			
			# If found, add object points, image points (after refining them)
			if corner_ret == True:
				
				cv2.imwrite("frame"+str(nframe)+".png", frame)

				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(gray, corners, grid_size, (-1,-1), criteria)
				imgpoints.append(corners2)
				img = cv2.drawChessboardCorners(img, grid_size, corners, corner_ret)

			# Draw and display the corners
			if imshow:
				cv2.imshow('gray',gray)
				cv2.imshow('img',img)

		else:

			pass

		if cv2.waitKey(delay) & 0xFF == 27:
			break

		

	cap.release()

	if imshow:
		cv2.destroyAllWindows()
		cv2.waitKey(1) 

	if len(objpoints) > 0 and len(imgpoints) > 0:
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		calib_params = {"ret" : ret, "mtx" : mtx, "dist" : dist, "rvecs" : rvecs, "tvecs" : tvecs}
	else:
		print "No calibration points found!"
		calib_params = None

	return calib_params


def get_calib_still(calib_images,  grid_size = (6,9), imshow = True, delay = 500):

	""" Returns calibration parameters from calibration images

		Parameters
		----------
		calib_images : list of strings
			List of file paths as strings (e.g. glob.glob("/path/to/files/*.jpg").
		grid_size : tuple of int
			Size of calibration grid (internal corners)
		imshow : bool
			Show the calibration images
		delay : int, >=1
			Delay in msecs between each image for imshow
			
		Returns
		-------
		calib_params : dict
			Parameters for undistorting images.

	"""

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((grid_size[0]*grid_size[1],3), np.float32)
	objp[:,:2] = np.mgrid[0:grid_size[1],0:grid_size[0]].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	for filename in calib_images:
		img = cv2.imread(filename)

		if img != None:

			#img = cv2.resize(img, (0,0), None, fx = 0.25, fy = 0.25)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			# Find the chess board corners
			ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

			# If found, add object points, image points (after refining them)
			if ret == True:
				objpoints.append(objp)
				corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
				imgpoints.append(corners2)

				if imshow == True:
					img = cv2.drawChessboardCorners(img, grid_size, corners2, ret)


			# Draw and display the corners
			if imshow == True:
				cv2.imshow('img',img)
				if cv2.waitKey(delay) & 0xFF == 27:
					break

	if imshow == True:
		cv2.destroyAllWindows()
		for i in range(5):
			cv2.waitKey(1) 
	if len(objpoints) > 0 and len(imgpoints) > 0:
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		calib_params = {"ret" : ret, "mtx" : mtx, "dist" : dist, "rvecs" : rvecs, "tvecs" : tvecs}
	else:
		print "No calibration points found!"
		calib_params = None

	return calib_params


def undistort_image(image, calib_params, crop = True):

	""" Returns undistorted image using calibration parameters.

		Parameters
		----------
		image : numpy_array 
			Image to be undistorted
		calib_params : dict
			Calibration parameters from calibrate_camera()
			
		Returns
		-------
		dst : numpy_array
			Undistorted image.

	"""
	try:
		ret = calib_params["ret"]
		mtx = calib_params["mtx"]
		dist = calib_params["dist"]
		rvecs = calib_params["rvecs"]
		tvecs = calib_params["tvecs"]
	except:
		raise TypeError("calib_params must be 'dict'")

	img = image
	h,  w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

	# undistort
	dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	# crop the image
	if crop:
		x,y,w,h = roi
		dst = dst[y:y+h, x:x+w]

	return dst

def save_calib(filename, calib_params): 
	""" Saves calibration parameters as '.pkl' file.

		Parameters
		----------
		filename : str
			Path to save file, must be '.pkl' extension
		calib_params : dict
			Calibration parameters to save

		Returns
		-------
		saved : bool
			Saved successfully.
	"""
	if type(calib_params) != dict:
			raise TypeError("calib_params must be 'dict'")

	output = open(filename, 'wb')

	try:
		pickle.dump(calib_params, output)
	except:
		raise IOError("File must be '.pkl' extension")

	output.close()

	saved = True

	return saved

def load_calib(filename):
	""" Loads calibration parameters from '.pkl' file.

		Parameters
		----------
		filename : str
			Path to load file, must be '.pkl' extension
			
		Returns
		-------
		calib_params : dict
			Parameters for undistorting images.

	"""
	# read python dict back from the file
	
	pkl_file = open(filename, 'rb')

	try:
		calib_params = pickle.load(pkl_file)
	except:
		raise IOError("File must be '.pkl' extension")

	pkl_file.close()

	return calib_params