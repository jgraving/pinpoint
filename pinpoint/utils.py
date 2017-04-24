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
from numba import jit
from scipy.spatial import distance as dist
from warnings import warn
from types import BooleanType, IntType, StringType, FloatType, NoneType, TupleType

def rotate_tag90(tag, tag_shape, n=1):
	
	"""Rotate barcode tag 90 degrees.
		
		Parameters
		----------
		tag : 1-D array_like
			Flattened barcode tag.
		tag_shape : tuple of int
			Shape of the barcode tag.
		n : int
			Number of times to rotate 90 degrees.
		
		
		Returns
		-------
		tag_rot : 1-D array
			Returns rotated tag flattened to 1-D array.
		
		"""
	
	tag = np.asarray(tag)
	vector_shape = tag.shape
	tag = tag.reshape(tag_shape)
	tag_rot = np.rot90(tag,n)
	tag_rot = tag_rot.reshape(vector_shape)
	return tag_rot

def add_border(tag, tag_shape, white_width = 1, black_width = 1):
	
	"""Add black and white border to barcode tag.
		
		Parameters
		----------
		tag : 1-D array_like
			Flattened barcode tag.
		tag_shape : tuple of int
			Shape of the barcode tag without a border.
		white_width : int
			Width of white border.
		black_width : int
			Width of black border.
			
		Returns
		-------
		bordered_tag : 1-D array
			Returns tag with border added flattened to 1-D array.
		"""
	
	tag = np.asarray(tag)
	tag = tag.reshape(tag_shape)

	black_border = np.zeros((tag_shape[0]+(2*white_width)+(2*black_width),tag_shape[1]+(2*white_width)+(2*black_width)))
	white_border = np.ones((tag_shape[0]+(2*white_width),tag_shape[1]+(2*white_width)))
	
	white_border[white_width:tag_shape[0]+white_width,white_width:tag_shape[1]+white_width] = tag
	black_border[black_width:tag_shape[0]+(2*white_width)+black_width, black_width:tag_shape[1]+(2*white_width)+black_width] = white_border

	tag = black_border
	bordered_tag = tag.reshape((1,tag.shape[0]*tag.shape[1]))
	tag_shape = black_border.shape
	return  tag_shape, bordered_tag

def check_diffs(array1, array2, ndiffs, test_num):
	
	""" Check for differences between two arrays. Each element of array1 is checked against all elements of array2.
		
		Parameters
		----------
		array1 : 2-D array_like
			Array of flattened barcodes to test against array2.
		array2 : 2-D array_like
			Array of flattened barcodes to test array1 against.
		ndiffs : int
			Minimum number of differences between all barcodes in array1 and array2.
		test_num : int
			Number of elements in array2 that each element in array1 must be at least ndiffs different from.
			
		Returns
		-------
		test : int
			Number of elements in array1 that are at least ndiffs different from all elements in array2.
		
		"""
	
	array1 = np.asarray(array1).astype(np.uint8)
	array2 = np.asarray(array2).astype(np.uint8)
	test_list = np.array([], dtype = bool)
	
	for tag in array1:
		tag = np.asarray([tag]).astype(np.uint8)
		repeat_tags = np.repeat(tag, array2.shape[0], axis = 0)
		tag_diffs = np.subtract(array2,repeat_tags)
		tag_diffs[tag_diffs>1] = 1
		tag_diffs = np.sum(tag_diffs, axis = 1)
		diff_bool = tag_diffs >= ndiffs # test for minimum number of differences
		test = np.sum(diff_bool)
			
		if test == test_num: # if element in array1 is different enough from all elements in array2...
			test_list = np.append(test_list, True)
	
	test = np.sum(test_list)
	
	return test

def crop(src, pt1, pt2):
	
	""" Returns a cropped version of src """
	
	cropped = src[pt1[1]:pt2[1], pt1[0]:pt2[0]]
	
	return cropped

def distance(vector):
 
	""" Return distance of vector """

	return np.sqrt(np.sum(np.square(vector)))

"""def order_points(pts):
	# sort the points based on their x-coordinates
	sorted_ = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-coordinate points
	left = sorted_[:2, :]
	right = sorted_[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	left = left[np.argsort(left[:, 1]), :]
	(tl, bl) = left
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], right, "euclidean")[0]
	(br, tr) = right[np.argsort(D)[::-1], :]
	
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")"""

@jit(nopython = True)
def _order_points(points):

	# Check for clockwise order with crossproduct
	dx1 = points[1,0] - points[0,0]
	dy1 = points[1,1] - points[0,1]
	dx2 = points[2,0] - points[0,0]
	dy2 = points[2,1] - points[0,1]
	cross_product = (dx1 * dy2) - (dy1 - dx2)

	if cross_product < 0.0: # if not clockwise, reorder middle points
		clockwise = points.copy()
		clockwise[1] = points[3]
		clockwise[3] = points[1]
	else:
		clockwise = points
		
	x_sorted = clockwise[np.argsort(clockwise[:, 0]), :] # sort by x-coords
	left = x_sorted[:2,:] # get left side coordinates
	left = left[np.argsort(left[:, 1]), :] # sort left coords by y-coords
	top_left = left[0] # get top left corner
	
	top_left_condition = (clockwise[:,0] == top_left[0]) & (clockwise[:,1] == top_left[1])
	top_left_index = np.where(top_left_condition)[0][0] #get original location for top left corner
	
	#reorder from the top left corner
	if top_left_index == 0:
		ordered = clockwise
	elif top_left_index == 1:
		ordered = clockwise.copy()
		ordered[0] = clockwise[1]
		ordered[1] = clockwise[2]
		ordered[2] = clockwise[3]
		ordered[3] = clockwise[0]
	elif top_left_index == 2:
		ordered = clockwise.copy()
		ordered[0] = clockwise[2]
		ordered[1] = clockwise[3]
		ordered[2] = clockwise[0]
		ordered[3] = clockwise[1]
	elif top_left_index == 3:
		ordered = clockwise.copy()
		ordered[0] = clockwise[3]
		ordered[1] = clockwise[0]
		ordered[2] = clockwise[1]
		ordered[3] = clockwise[2]

	return ordered
	
def order_points(points):

	""" Order 4x2 array in clockwise order from top-left corner.

		Parameters
		----------
		points : array_like
			4x2 array of points to order.
			
		Returns
		-------
		ordered : ndarray
			Points ordered clockwise from top-left corner.
		
		"""
	if type(points) != np.ndarray:
		raise TypeError("points must be numpy array")
	if points.ndim != 2:
		raise ValueError("points must be 2-D array")
	if points.shape != (4,2):
		raise ValueError("points must be 4x2 array")
	if points.dtype != np.dtype(np.float32):
		points = points.astype(np.float32)
		
	ordered = _order_points(points)
	
	return ordered
	
def rowwise_corr(A,B):
	# Rowwise mean of input arrays & subtract from input arrays themeselves
	A_mA = np.subtract(A, A.mean(1)[:,None])
	B_mB = np.subtract(B, B.mean(1)[:,None])

	# Sum of squares across rows
	ssA = np.square(A_mA).sum(1)
	ssB = np.square(B_mB).sum(1)

	# Finally get corr coeff
	return np.divide(np.dot(A_mA,B_mB.T), np.sqrt(np.dot(ssA[:,None],ssB[None])))

def angle(vector, degrees = True):
	
	"""Returns the angle between vectors 'v1' and 'v2'.
		
		Parameters
		----------
		v1 : 1-D array_like
			N-dimensional vector.
		v2 : 1-D array_like
			N-dimensional vector.
		degrees : bool, default = True
			Return angle in degrees.
			
		Returns
		-------
		angle : float
			Angle between v1 and v2.
		
		"""
	
	angle = np.arctan2(vector[1], vector[0]) % (2*np.pi)
	if np.isnan(angle):
		if (v1_u == v2_u).all():
			return 0.0
		else:
			return np.pi
	if degrees == True:
		angle = np.degrees(angle)
	return angle

def get_grayscale(color_image, channel = None):

	""" Returns single-channel grayscale image from 3-channel BGR color image.

		Parameters
		----------
		color_image : (MxNx3) numpy array
			3-channel BGR-format color image as a numpy array
		channel : {'blue', 'green', 'red', 'none', None}, default = None
			The color channel to use for producing the grayscale image.
			
		Returns
		-------
		gray_image : (MxNx1) numpy array
			Single-channel grayscale image as a numpy array.

		Notes
		----------
		For channel, None uses the default linear transformation from OpenCV: Y = 0.299R + 0.587G + 0.114B
		Channels 'blue', 'green', and 'red' use the respective color channel as the grayscale image. 
		Under white ambient lighting, 'green' typically provides the lowest noise. Under red and infrared lighting, 'red' typically provides the lowest noise.

	"""
	if type(channel) not in [StringType, NoneType]:
		raise TypeError("Channel must be type str or None")
	if type(channel) is StringType and not (channel.startswith('b') or channel.startswith('g') or channel.startswith('r')):
		raise ValueError("Channel value must be 'blue', 'green', 'red', or None")
	if type(color_image) is not np.ndarray:
		raise TypeError("image must be a numpy array")


	if len(color_image.shape) != 3:
		raise ValueError("image must be color (MxNx3)")
	if color_image.shape[2] != 3:
		raise ValueError("image must have 3 color channels (MxNx3)")
	if color_image.dtype is not np.dtype(np.uint8):
		raise TypeError("image array must be dtype np.uint8")

	if channel.startswith('b'):
		gray_image, _, _ = cv2.split(color_image)
	elif channel.startswith('g'):
			_, gray_image, _ = cv2.split(color_image)
	elif channel.startswith('r'):
		_, _, gray_image = cv2.split(color_image)
	elif channel is None:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

	return gray_image

def get_threshold(gray_image, block_size = 1001, offset = 2):

	""" Returns binarized thresholded image from single-channel grayscale image.

		Parameters
		----------
		gray_image : (MxNx1) numpy array
			Single-channel grayscale image as a numpy array
		block_size : int, default = 1001
			Odd value integer. Size of the local neighborhood for adaptive thresholding.
		offset : default = 2
			Constant subtracted from the mean. Normally, it is positive but may be zero or negative as well. 
			The threshold value is the mean of the block_size x block_size neighborhood minus offset.

		Returns
		-------
		threshold_image : (MxNx1) numpy array
			Binarized (0, 255) image as a numpy array.

	"""

	if block_size % 2 != 1:
		raise ValueError("block_size must be an odd value (block_size % 2 == 1)")
	if type(offset) not in [IntType, FloatType]:
		raise TypeError("offset must be type int or float")
	if type(gray_image) is not np.ndarray:
		raise TypeError("image must be a numpy array")
	if len(gray_image.shape) != 2:
		raise ValueError("image must be grayscale")
	if gray_image.dtype is not np.dtype(np.uint8):
		raise TypeError("image array must be dtype np.uint8")

	threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, offset)

	return threshold_image

def get_contours(threshold_image):

	""" Returns a list of contours from a binarized thresholded image.

		Parameters
		----------
		threshold_image : (MxNx1) numpy array
			Binarized threshold image as a numpy array

		Returns
		-------
		contours : list
			List of contours extracted from threshold_image.

	"""
	if len(set([0, 255]) - set(np.unique(threshold_image))) != 0:
		raise ValueError("image must be binarized to (0, 255)")

	_, contours, _ = cv2.findContours(threshold_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return contours

def get_polygon(contour, tolerance):

	perimeter = cv2.arcLength(contour, True)
	polygon = cv2.approxPolyDP(contour, tolerance * perimeter, True)
	polygon_area = cv2.contourArea(polygon, True)

	return (polygon, polygon_area, perimeter)

def test_edge_proximity(contour, edge_proximity, x_proximity, y_proximity):

	contour_reshape = np.squeeze(contour)
	contour_x = contour_reshape[:,0]
	contour_y = contour_reshape[:,1]
	flat = contour.flatten()

	edge_zero = np.sum(flat <= edge_proximity)
	edge_x = np.sum(contour_x >= x_proximity)
	edge_y = np.sum(contour_y >= y_proximity)

	if edge_zero == 0 and edge_x == 0 and edge_y == 0:
		edge_proximity_test = False
	else:
		edge_proximity_test = True

	return edge_proximity_test

def get_pixels(image, points, dst, max_side, barcode_size):

	M = cv2.getPerspectiveTransform(points, dst)
	warped = cv2.warpPerspective(image, M, (max_side, max_side), flags = (cv2.INTER_LINEAR), borderValue = 255)
	pixels = pixels = cv2.resize(warped, barcode_size, interpolation = cv2.INTER_AREA)

	return pixels

def get_points(polygon):

	points = np.squeeze(polygon).astype(np.float32)
	points = order_points(points)

	return points

def test_area(area, area_min, area_max, area_sign):
	if np.sign(area) == area_sign and area_min <= np.abs(area) <= area_max:
		area_test = True
	else:
		area_test = False

	return area_test

def test_quad(polygon, polygon_area, area_min, area_max, area_sign):

	# if 4 vertices, sign is correct, value is within range, and polygon is convex...
	if (polygon.shape[0] == 4 
		and test_area(polygon_area, area_min, area_max, area_sign)
		and cv2.isContourConvex(polygon)):

		quad_test = True

	else:
		quad_test = False

	return quad_test

def get_area(contour):

	return cv2.contourArea(contour, True)

def test_geometry(contour, area_min, area_max, area_sign, edge_proximity, x_proximity, y_proximity, tolerance):
	
	geometry_test = False
	polygon = None

	if contour.shape[0] > 4:

		edge_proximity_test = test_edge_proximity(contour, edge_proximity, x_proximity, y_proximity)

		if edge_proximity_test == False:

			contour_area = get_area(contour) # calculate the signed area

			if test_area(contour_area, area_min, area_max, area_sign): # if the sign is correct and value is within range...
				
				(polygon, polygon_area, perimeter) = get_polygon(contour, tolerance)
				peri_area_ratio = perimeter/contour_area

				if -100 < peri_area_ratio < 0:  # if perimeter_area_ratio is reasonable value...

					geometry_test = test_quad(polygon, polygon_area, area_min, area_max, area_sign)

	return (geometry_test, polygon)


def get_candidate_barcodes(image, contours, barcode_size, area_min, area_max, area_sign, edge_proximity, x_proximity, y_proximity, tolerance, max_side, dst):

	FIRST = True
	points_array = None
	pixels_array = None
	points = None
	pixels = None

	if len(contours) > 0: # if contours are found
					
		for (idx,contour) in enumerate(contours): # iterate over the list of contours

			(geometry_test, polygon) = test_geometry(contour, area_min, area_max, area_sign, edge_proximity, x_proximity, y_proximity, tolerance)
			
			if geometry_test == True:
				points = get_points(polygon).reshape((1,4,2))
				pixels = get_pixels(image, points, dst, max_side, barcode_size).reshape((1,-1))

				if FIRST == True:
					points_array = points
					pixels_array = pixels
					FIRST = False
				else:
					points_array = np.append(points_array, points, axis = 0)
					pixels_array = np.append(pixels_array, pixels, axis = 0) 

	return (points, pixels, points_array, pixels_array)

def correlate_barcodes(pixels, master_list):

	return rowwise_corr(pixels, master_list)

def match_barcodes(points_array, pixels_array, master_list, IDs, IDs_index, correlation_thresh):

	master_correlation_matrix = correlate_barcodes(pixels_array, master_list)
	best_values_index = np.argmax(master_correlation_matrix, axis = 0)
	best_values = np.max(master_correlation_matrix, axis = 0)

	IDs_index = IDs_index[best_values > correlation_thresh]

	best_values_index = best_values_index[best_values > correlation_thresh]
	best_IDs = IDs[best_values > correlation_thresh]
	best_points = points_array[best_values_index]

	if best_value > 0.8:

		best_index = np.argmax(correlation)
		ID = IDs[best_index]
		rotation_test = best_index % 4

		corners = np.zeros((4,2))

		if rotate_test == 3:
			corners = points

		if rotate_test == 0:
			corners[0] = bl
			corners[1] = tl
			corners[2] = tr
			corners[3] = br
			
		if rotate_test == 1:
			corners[0] = br
			corners[1] = bl
			corners[2] = tl
			corners[3] = tr
			
		if rotate_test == 2:
			corners[0] = tr
			corners[1] = br
			corners[2] = bl
			corners[3] = tl

		edge = np.array(np.mean([corners[0], corners[1]], axis = 0))
		centroid = np.mean(corners, axis = 0)
		corners[:,1] = -corners[:,1]
		edge[1] = -edge[1]
		centroid[1] = -centroid[1]
		vector = np.subtract(edge, centroid)
		vector_angle = utils.angle(vector)

def get_warp_dst(max_side = 100):
	
	length = max_side - 1
	dst = np.array([[0, 0],
					[length, 0],
					[length, length],
					[0, length]],
					dtype = "float32")

	return dst


