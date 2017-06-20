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
from warnings import warn

from sklearn.metrics.pairwise import pairwise_distances


# disable multithreading in OpenCV for main thread 
# to avoid problems after parallelization
cv2.setNumThreads(0)


def rotate_tag90(tag, tag_shape, n=1):
	
	"""
	Rotate barcode tag 90 degrees.
	
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
	
	"""
	Add black and white border to barcode tag.
		
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
	bordered_tag = tag.reshape((tag.shape[0]*tag.shape[1]))
	tag_shape = black_border.shape
	return  bordered_tag

def crop(src, pt1, pt2):
	
	""" Returns a cropped version of src """
	
	cropped = src[pt1[1]:pt2[1], pt1[0]:pt2[0]]
	
	return cropped


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

	""" 
	Order 4x2 array in clockwise order from top-left corner.

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

def get_grayscale(color_image, channel = None):

	""" 
	Returns single-channel grayscale image from 3-channel BGR color image.

	Parameters
	----------
	color_image : array_like, shape (MxNx3)
		3-channel BGR-format color image
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
	if type(channel) not in [str, type(None)]:
		raise TypeError("Channel must be type str or None")
	if type(channel) is str:
		if not (channel.startswith('b') or channel.startswith('g') or channel.startswith('r')):
			raise ValueError("Channel value must be 'blue', 'green', 'red', or None")
	if type(color_image) is not np.ndarray:
		raise TypeError("image must be a numpy array")


	if color_image.ndim != 3:
		raise ValueError("image must be color (MxNx3)")
	if color_image.shape[2] != 3:
		raise ValueError("image must have 3 color channels (MxNx3)")
	#if color_image.dtype is not np.dtype(np.uint8):
#		raise TypeError("image array must be dtype np.uint8")

	if channel.startswith('b'):
		gray_image, _, _ = cv2.split(color_image)
	elif channel.startswith('g'):
			_, gray_image, _ = cv2.split(color_image)
	elif channel.startswith('r'):
		_, _, gray_image = cv2.split(color_image)
	elif channel is None:
		gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

	return gray_image

def get_threshold(gray_image, block_size = 101, offset = 0):

	""" 
	Returns binarized thresholded image from single-channel grayscale image.

	Parameters
	----------
	gray_image : array_like, shape  (MxNx1)
		Single-channel grayscale image
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
	if type(offset) not in [int, float]:
		raise TypeError("offset must be type int or float")
	if type(gray_image) is not np.ndarray:
		raise TypeError("image must be a numpy array")
	if gray_image.ndim != 2:
		raise ValueError("image must be grayscale")
	#if gray_image.dtype is not np.dtype(np.uint8):
	#	raise TypeError("image array must be dtype np.uint8")

	threshold_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, offset)

	return threshold_image

def get_contours(threshold_image):

	""" 
	Returns a list of contours from a binarized thresholded image.

	Parameters
	----------
	threshold_image : (MxNx1) numpy array
		Binarized threshold image as a numpy array
	gray_image : array_like, shape  (MxNx1)
		Binarized threshold image
	Returns
	-------
	contours : list
		List of contours extracted from threshold_image.

	"""
	#if len(set([0, 255]) - set(np.unique(threshold_image))) != 0:
		#raise ValueError("image must be binarized to (0, 255)")

	_, contours, _ = cv2.findContours(threshold_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	return contours

def get_polygon(contour, tolerance=0.1):

	""" 
	Returns a polygonal approximation of a contour.

	Parameters
	----------
	contour : array_like
		Contour to fit a polygon to
	tolerance : int or float, default = 0.1
		Tolerance for fitting a polygon as a proportion of the perimeter of the contour. 
		This value is used to set epsilon, which is the maximum distance between the original contour 
		and its polygon approximation. Higher values decrease the number of vertices in the polygon.
		Lower values increase the number of vertices in the polygon. This parameter affects 
		how many many contours reach the barcode matching algorithm, 
		as only polygons with 4 vertices are used.
	Returns
	-------
	polygon : ndarray
		The fitted polygon.
	polygon_area : float
		The signed area of the polygon
	perimeter : float
		The perimeter of the contour
		

	"""
	perimeter = cv2.arcLength(contour, True)
	polygon = cv2.approxPolyDP(contour, tolerance * perimeter, True)
	polygon_area = cv2.contourArea(polygon, True)

	return (polygon, polygon_area, perimeter)

def test_edge_proximity(contour, edge_proximity, x_proximity, y_proximity):

	""" 
	Test if a contour is too close to the edge of the frame to read.

	Parameters
	----------
	contour : array_like
		Contour to test
	edge_proximity : int, default = 1
		The threshold in pixels for how close a contour can be to the edge. 
		Default is 1 pixel
	x_proximity : int
		The threshold in pixels for how close a contour can be to the x-axis border.
		This should correspond to frame_width - edge_proximity, but is precalculated for speed
	y_proximity : int 
		The threshold in pixels for how close a contour can be to the y-axis border.
		This should correspond to frame_height - edge_proximity, but is precalculated for speed

	Returns
	-------
	edge_proximity_test : bool
		Returns False if the contour is far enough away from the edge of the frame

	"""
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

def get_pixels(image, points, template, max_side, barcode_shape):

	""" 
	Use affine transform to extract pixel values of candidate barcodes from an image.

	Parameters
	----------
	image : array_like
		Grayscale image from which to extract pixel values
	points : array_like
		An array of coordinates corresponding to the four corners of the candidate barcode
	template : array_like
		A template for storing the extracted pixels. 
		This should be precalculated using get_tag_template()
	max_side : int 
		The size of the template in pixels

	Returns
	-------
	pixels : ndarray
		The extracted pixel values

	"""
	M = cv2.getPerspectiveTransform(points.reshape((4,1,2)).astype(np.float32), template)
	pixels = cv2.warpPerspective(image, M, (max_side, max_side), flags = (cv2.INTER_LINEAR), borderValue = 255)
	pixels = cv2.resize(pixels, barcode_shape)

	return pixels

def get_points(polygon):
	""" 
	Sort corners of a candidate barcode clockwise from the top-left corner.

	Parameters
	----------
	polygon : array_like, shape (4,1,2)
		A polygon with four coordinates to sort

	Returns
	-------
	points : ndarray
		The coordinates sorted clockwise from the top-left corner.

	"""

	polygon = np.squeeze(polygon)
	points = order_points(polygon)
	points = polygon.reshape((1,4,2))

	return points

def test_area(area, area_min, area_max, area_sign):

	"""
	Test the area of a candidate barcode.

	Parameters
	----------
	area : float
		The area of the candidate barcode
	area_min : float
		Minimum area
	area_max : float
		Maximum area
	area_sign : +1 or -1
		The sign of the area

	Returns
	-------
	area_test : bool
		Returns True if sign is correct and area is within range.

	"""
	if np.sign(area) == area_sign and area_min <= np.abs(area) <= area_max:
		area_test = True
	else:
		area_test = False

	return area_test

def test_quad(polygon, polygon_area, area_min, area_max, area_sign):
	"""
	Test if a polygon is a quadrilateral.

	Parameters
	----------
	polygon : array_like, shape (4,1,2)
		A polygon to test
	polygon_area : float
		The area of the candidate barcode
	area_min : float
		Minimum area
	area_max : float
		Maximum area
	area_sign : +1 or -1
		The sign of the area

	Returns
	-------
	quad_test : bool
		Returns True if the polygon is a quadrilateral.

	"""

	# if 4 vertices, sign is correct, value is within range, and polygon is convex...
	if (polygon.shape[0] == 4 
		and test_area(polygon_area, area_min, area_max, area_sign)
		and cv2.isContourConvex(polygon)):

		quad_test = True

	else:
		quad_test = False

	return quad_test

def get_area(contour):

	""" 
	Calculate the signed area of a contour.

	Parameters
	----------
	contour : array_like
		Contour to calculate the area for

	Returns
	-------
	area : int
		The signed area of a contour
	"""
	area = cv2.contourArea(contour, True)
	
	return area

def test_geometry(contour, area_min, area_max, area_sign, edge_proximity, x_proximity, y_proximity, tolerance):

	"""
	Test the geometry of a contour. Return the corners if it is a candidate barcode.

	Parameters
	----------
	contour : array_like
		The contour to test
	area_min : float
		Minimum area
	area_max : float
		Maximum area
	area_sign : +1 or -1
		The sign of the area
	edge_proximity : int, default = 1
		The threshold in pixels for how close a contour can be to the edge. 
		Default is 1 pixel
	x_proximity : int
		The threshold in pixels for how close a contour can be to the x-axis border.
		This should correspond to frame_width - edge_proximity, but is precalculated for speed
	y_proximity : int 
		The threshold in pixels for how close a contour can be to the y-axis border.
		This should correspond to frame_height - edge_proximity, but is precalculated for speed
	tolerance : int or float, default = 0.1
		Tolerance for fitting a polygon as a proportion of the perimeter of the contour. 
		This value is used to set epsilon, which is the maximum distance between the original contour 
		and its polygon approximation. Higher values decrease the number of vertices in the polygon.
		Lower values increase the number of vertices in the polygon. This parameter affects 
		how many many contours reach the barcode matching algorithm, 
		as only polygons with 4 vertices are used.
		
	Returns
	-------
	geometry_test : bool
		Returns True if the contour is a candidate barcode.
	polygon : ndarray
		The polygon approximation of the contour
	"""

	geometry_test = False
	polygon = np.zeros((0,4,2))

	if contour.shape[0] >= 4:

		edge_proximity_test = test_edge_proximity(contour, edge_proximity, x_proximity, y_proximity)

		if edge_proximity_test == False:

			contour_area = get_area(contour) # calculate the signed area

			if test_area(contour_area, area_min, area_max, area_sign): # if the sign is correct and value is within range...
				
				(polygon, polygon_area, perimeter) = get_polygon(contour, tolerance)
				peri_area_ratio = perimeter/contour_area

				if 1 > np.abs(peri_area_ratio) > 0:  # if perimeter_area_ratio is reasonable value...

					geometry_test = test_quad(polygon, polygon_area, area_min, area_max, area_sign)

	return (geometry_test, polygon)


def get_candidate_barcodes(image, contours, barcode_shape, area_min, area_max, area_sign, edge_proximity, x_proximity, y_proximity, tolerance, template, max_side):

	"""
	Find candidate barcodes in an image.

	Parameters
	----------
	image : array_like, shape (n_rows, n_cols, 1)
		An image containing barcodes
	contours : list
		A list of contours from cv2.contours()
	barcode_shape : tuple
		The shape of the barcode with white border
	area_min : float
		Minimum area
	area_max : float
		Maximum area
	area_sign : +1 or -1
		The sign of the area
	edge_proximity : int, default = 1
		The threshold in pixels for how close a contour can be to the edge. 
		Default is 1 pixel
	x_proximity : int
		The threshold in pixels for how close a contour can be to the x-axis border.
		This should correspond to frame_width - edge_proximity, but is precalculated for speed
	y_proximity : int 
		The threshold in pixels for how close a contour can be to the y-axis border.
		This should correspond to frame_height - edge_proximity, but is precalculated for speed
	tolerance : int or float, default = 0.1
		Tolerance for fitting a polygon as a proportion of the perimeter of the contour. 
		This value is used to set epsilon, which is the maximum distance between the original contour 
		and its polygon approximation. Higher values decrease the number of vertices in the polygon.
		Lower values increase the number of vertices in the polygon. This parameter affects 
		how many many contours reach the barcode matching algorithm, as only polygons with 4 vertices are used.
	template : array_like
		A template for storing the extracted pixels. 
		This should be precalculated using get_tag_template()
	max_side : int 
		The size of the template in pixels

	Returns
	-------
	points_array : ndarray, shape (n_samples, 4, 2)
		Array of coordinates for candidate barcodes.
	pixels_array : ndarray, shape (n_samples, barcode_shape[0]*barcode_shape[1])
		Array of flattened pixels for candidate barcodes
	"""

	points_array = np.zeros((0, 4, 2), dtype=np.int32)
	pixels_array = np.zeros((0, barcode_shape[0]*barcode_shape[1]), dtype=np.uint8)

	if len(contours) > 0: # if contours are found
					
		for (idx,contour) in enumerate(contours): # iterate over the list of contours

			(geometry_test, polygon) = test_geometry(contour, area_min, area_max, area_sign, edge_proximity, x_proximity, y_proximity, tolerance)
			
			if geometry_test == True:
				points = get_points(polygon)
				pixels = get_pixels(image, points, template, max_side, barcode_shape).reshape((1,-1))

				points_array = np.append(points_array, points, axis = 0)
				pixels_array = np.append(pixels_array, pixels, axis = 0)
	subpix = points_array.reshape((points_array.shape[0]*points_array.shape[1], 1, points_array.shape[2])).astype(np.float32)
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.0001)
	subpix = cv2.cornerSubPix(image, subpix, (5,5), (-1,-1), criteria)
	points_array = subpix.reshape(points_array.shape)
	return (points_array, pixels_array)

def match_barcodes(points_array, pixels_array, barcode_nn, id_list, id_index, distance_threshold):
	
	"""
	Match candidate barcodes

	Parameters
	----------
	points_array : array_like, shape (n_samples, 4, 2)
		Array of coordinates for candidate barcodes
	pixels_array : array_like, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes
	barcode_nn : NearestNeighbors class
		A NearestNeighbors class from sci-kit learn fitted to a list of barcodes.
		See sklearn.neighbors.NearestNeighbors for details
	id_list : array_like
		An array of known identities corresponding to the list of barcodes in barcode_nn
	id_index : array_like
		An array of indices corresponding to id_list
	distance_threshold : float
		The maximum distance between a barcode candidate and its matched identity


	Returns
	-------
	points_array : ndarray, shape (n_samples, 4, 2)
		Array of coordinates for matched barcodes.
	pixels_array : ndarray, shape (n_samples, barcode_shape[0]*barcode_shape[1])
		Array of flattened pixels for matched barcodes
	best_id_list : ndarray
		Array of identities for matched barcodes
	best_id_index : ndarray
		Array of indices corresponding to best_id_list
	distances : ndarray
		Array of distances for matched barcodes
	"""
	
	distances, index = barcode_nn.kneighbors(pixels_array//255, n_neighbors=1)
	distances = distances[:,0]
	index = index[:,0]
	
	pixel_index = np.where(distances < distance_threshold)
	distances = distances[pixel_index]
	index = index[pixel_index]

	points_array = points_array[pixel_index]
	pixels_array = pixels_array[pixel_index]

	best_id_list = id_list[index]
	best_id_index = id_index[index]

	return (points_array, pixels_array, best_id_list, best_id_index, distances)

@jit(nopython=True)
def sort_corners(points_array, best_id_index):
	"""
	Sort coordinates of matched barcode corners in clockwise order from the true top-left corner
	(taking into account the orientation of the tag).

	Parameters
	----------
	points_array : array_like, shape (n_samples, 4, 2)
		Array of coordinates for candidate barcodes
	best_id_index : ndarray
		Array of indices corresponding to best_id_list

	Returns
	-------
	points_array : ndarray, shape (n_samples, 4, 2)
		Array of sorted coordinates for matched barcodes
	"""

	for idx in range(points_array.shape[0]):
		
		points = points_array[idx]
		best_index = best_id_index[idx]
		rotation_test = best_index % 4
		
		tl = points[0]
		tr = points[1]
		br = points[2]
		bl = points[3]
		
		corners = np.empty_like(points)

		if rotation_test == 3:
			corners = points

		if rotation_test == 0:
			corners[0] = bl
			corners[1] = tl
			corners[2] = tr
			corners[3] = br

		if rotation_test == 1:
			corners[0] = br
			corners[1] = bl
			corners[2] = tl
			corners[3] = tr

		if rotation_test == 2:
			corners[0] = tr
			corners[1] = br
			corners[2] = bl
			corners[3] = tl

		points_array[idx] = corners

	return points_array

def get_tag_template(max_side):
	"""
	Create a template for extracting barcode pixels using affine transform.
	
	Parameters
	----------
	max_side : int 
		The size of the template in pixels
	Returns
	-------
	template : array_like
		A template for storing extracted barcode pixels. 
	
	"""
	
	length = max_side - 1
	template = np.array([[0, 0],
					[length, 0],
					[length, length],
					[0, length]],
					dtype = "float32")

	return template

def test_pixel_variance(points_array, pixels_array, var_thresh=500):
	"""
	Test pixel variance of each barcode and return only those with variance higher than a set threshold.
	Low variance typically means the candidate barcode is a white or black blob and not an actual barcode.
	
	Parameters
	----------
	points_array : ndarray, shape (n_samples, 4, 2)
		Array of coordinates for candidate barcodes
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes
	var_thresh : float, (default = 500)
		Minimum variance threshold
			
	Returns
	-------
	points_array : ndarray, shape (n_samples, 4, 2)
		Array of coordinates for candidate barcodes with variance higher than var_thresh
	pixels_array : ndarray, shape (n_samples, 4, 2)
		Array of flattened pixels for candidate barcodes with variance higher than var_thresh
	"""

	variances = np.var(pixels_array, axis=1)
	variance_index = np.where(variances >= var_thresh)
	pixels_array = pixels_array[variance_index]
	points_array = points_array[variance_index]

	return (points_array, pixels_array)

def otsu_threshold(pixels_array):
	"""
	Threshold barcode pixels using Otsu's method. The algorithm assumes the image
	contains two classes of pixels following a bimodal histogram. It calculates the
	optimum threshold separating the two classes so that their combined intra-class variance
	is minimal and their inter-class variance is maximal. 
	
	Parameters
	----------
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes
			
	Returns
	-------
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes binarized to 0,255
	"""
	for idx,pixels in enumerate(pixels_array):
		ret, pixels = cv2.threshold(pixels,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		pixels = np.squeeze(pixels)
		pixels_array[idx] = pixels
		
	return pixels_array

def whiten_edge_pixels(pixels_array, barcode_shape, white_width):
	"""
	Replace edge around the barcode pixels with white pixels for matching.
	
	Parameters
	----------
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes
	barcode_shape : tuple
		The shape of the barcode with white border
	white_width : int
		The bit width of the white border around the barcode
	Returns
	-------
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes with white edges
	"""
	pixels_array = pixels_array.reshape((pixels_array.shape[0], barcode_shape[0], barcode_shape[1]))
	edge_array = np.ones_like(pixels_array)
	edge_array[:, white_width:-white_width, white_width:-white_width] = 0
	pixels_array[edge_array == 1] = 255
	pixels_array = pixels_array.reshape((pixels_array.shape[0], barcode_shape[0]*barcode_shape[1]))
	
	return pixels_array

def preprocess_pixels(points_array, pixels_array, var_thresh, barcode_shape, white_width):
	"""
	Preprocess candidate barcode pixels before matching. 
	Test pixel variance, binarize the pixel values, and whiten the edge pixels for each barcode
	
	Parameters
	----------
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes
			
	Returns
	-------
	points_array : ndarray, shape (n_samples, 4, 2)
		Array of coordinates for candidate barcodes
	pixels_array : ndarray, shape (n_samples, n_pixels)
		Array of flattened pixels for candidate barcodes with preprocessed pixels
	"""
	points_array, pixels_array = test_pixel_variance(points_array, pixels_array, var_thresh)
	if pixels_array.shape[0] > 0:
		pixels_array = otsu_threshold(pixels_array)
	if pixels_array.shape[0] > 0:
		pixels_array = whiten_edge_pixels(pixels_array, barcode_shape, white_width)

	return (points_array, pixels_array)

def process_frame(frame, channel, resize, block_size, offset, barcode_shape, white_width, area_min, area_max, x_proximity, y_proximity, tolerance, template, max_side, var_thresh, barcode_nn, id_list, id_index, distance_threshold):
	"""
	Process a single frame to track barcodes. 
	
	Parameters
	----------
	frame : array_like, shape (MxNx3)
		3-channel BGR-format color image
	channel : {'blue', 'green', 'red', 'none', None}, default = None
		The color channel to use for producing the grayscale image.
	block_size : int, default = 1001
		Odd value integer. Size of the local neighborhood for adaptive thresholding.
	offset : default = 2
		Constant subtracted from the mean. Normally, it is positive but may be zero or negative as well. 
		The threshold value is the mean of the block_size x block_size neighborhood minus offset.
	barcode_shape : tuple
		The shape of the barcode with white border
	white_width : int
		The bit width of the white border around the barcode
	area_min : float
		Minimum area
	area_max : float
		Maximum area
	x_proximity : int
		The threshold in pixels for how close a contour can be to the x-axis border.
		This should correspond to frame_width - 1, but is precalculated for speed
	y_proximity : int 
		The threshold in pixels for how close a contour can be to the y-axis border.
		This should correspond to frame_height - 1, but is precalculated for speed
	tolerance : int or float, default = 0.1
		Tolerance for fitting a polygon as a proportion of the perimeter of the contour. 
		This value is used to set epsilon, which is the maximum distance between the original contour 
		and its polygon approximation. Higher values decrease the number of vertices in the polygon.
		Lower values increase the number of vertices in the polygon. This parameter affects 
		how many many contours reach the barcode matching algorithm, 
		as only polygons with 4 vertices are used.
	template : array_like
		A template for storing the extracted pixels. 
		This should be precalculated using get_tag_template()
	max_side : int 
		The size of the template in pixels
	var_thresh : float, (default = 500)
		Minimum variance threshold
	barcode_nn : NearestNeighbors class
		A NearestNeighbors class from sci-kit learn fitted to a list of barcodes.
		See sklearn.neighbors.NearestNeighbors for details
	id_list : array_like
		An array of known identities corresponding to the list of barcodes in barcode_nn
	id_index : array_like
		An array of indices corresponding to id_list
	distance_threshold : float
		The maximum distance between a barcode candidate and its matched identity

	Returns
	-------
	fetch_dict : dict

		A dictionary containing the following objects:

		"gray" : ndarray, shape (MxNx1)
			The grayscale image
		"thresh" : ndarray, shape (MxNx1)
			The threshold image
		"points_array" : ndarray, shape (n_samples, 4, 2)
			Array of coordinates for barcodes
		"pixels_array" : ndarray, shape (n_samples, n_pixels)
			Array of flattened pixels for barcodes
		"best_id_list" : ndarray, shape (n_samples)
			Array of identities that best match each barcode
		"distances" : ndarray, shape (n_samples)
			Array of Hamming distances between each barcode 
			and the closest match
	"""
	best_id_list = np.zeros((0,1))
	distances = np.zeros((0,1))
	
	gray = get_grayscale(frame, channel=channel)
	if resize > 1:
		gray = cv2.resize(gray, (0,0), None, resize, resize)
	gray = np.flipud(gray)
	thresh = get_threshold(gray, block_size=block_size, offset=offset)
	contours = get_contours(thresh)

	points_array, pixels_array = get_candidate_barcodes(image=gray,
														contours=contours,
														barcode_shape=barcode_shape,
														area_min=area_min,
														area_max=area_max,
														area_sign=-1,
														edge_proximity=1,
														x_proximity=x_proximity,
														y_proximity=y_proximity,
														tolerance=tolerance,
														max_side=max_side,
														template=template
													   )
	
	if pixels_array.shape[0] > 0:
		points_array, pixels_array = preprocess_pixels(points_array=points_array,
														pixels_array=pixels_array,
														var_thresh=var_thresh,
														barcode_shape=barcode_shape,
														white_width=white_width
														)
	if pixels_array.shape[0] > 0:
		points_array, pixels_array, best_id_list, best_id_index, distances = match_barcodes(points_array=points_array,
																  pixels_array=pixels_array,
																  barcode_nn=barcode_nn,
																  id_list=id_list,
																  id_index=id_index,
																  distance_threshold=distance_threshold
																						   )
	if points_array.shape[0] > 0:                                                           
		points_array = sort_corners(points_array, best_id_index)
	
	fetch_dict = {"gray":gray,
				"thresh":thresh,
				"points_array":points_array,
				"pixels_array":pixels_array,
				"best_id_list":best_id_list,
				"distances":distances}

	return fetch_dict