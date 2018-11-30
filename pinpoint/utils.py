"""
Copyright 2015-2018 Jacob M. Graving <jgraving@gmail.com>

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

# disable multithreading in OpenCV for main thread
# to avoid problems after parallelization
cv2.setNumThreads(0)


def crop(src, pt1, pt2):

    """ Returns a cropped version of src """

    cropped = src[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    return cropped


@jit(nopython=True)
def _order_points(points):

    # Check for clockwise order with crossproduct
    dx1 = points[1, 0] - points[0, 0]
    dy1 = points[1, 1] - points[0, 1]
    dx2 = points[2, 0] - points[0, 0]
    dy2 = points[2, 1] - points[0, 1]
    cross_product = (dx1 * dy2) - (dy1 - dx2)

    if cross_product < 0.0:  # if not clockwise, reorder middle points
        clockwise = points.copy()
        clockwise[1] = points[3]
        clockwise[3] = points[1]
    else:
        clockwise = points

    x_sorted = clockwise[np.argsort(clockwise[:, 0]), :]  # sort by x-coords
    left = x_sorted[:2, :]  # get left side coordinates
    left = left[np.argsort(left[:, 1]), :]  # sort left coords by y-coords
    top_left = left[0]  # get top left corner

    # get original location for top left corner
    top_left_condition = (clockwise[:, 0] == top_left[0]) & \
        (clockwise[:, 1] == top_left[1])
    top_left_index = np.where(top_left_condition)[0][0]

    # reorder from the top left corner
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


def grayscale(color_image, channel=None):

    """
    Returns single-channel grayscale image from 3-channel BGR color image.

    Parameters
    ----------
    color_image : array_like, shape (MxNx3)
        3-channel BGR-format color image
    channel : {0, 1, 2, None}, default = None
        The color channel to use for producing the grayscale image.
    Returns
    -------
    gray_image : (MxNx1) numpy array
        Single-channel grayscale image as a numpy array.

    """
    if channel is not None:
        gray_image = color_image[..., channel]
    else:
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    return gray_image


def adaptive_threshold(gray_image, block_size=101, offset=0):

    """
    Returns binarized thresholded image from single-channel grayscale image.

    Parameters
    ----------
    gray_image : array_like, shape (M, N, 1)
        Single-channel grayscale image
    block_size : int, default = 1001
        Odd value integer. Size of the local neighborhood
        for adaptive thresholding.
    offset : default = 2
        Constant subtracted from the mean. Normally, it is positive
        but may be zero or negative as well. The threshold value is
        the mean of the block_size x block_size neighborhood minus offset.

    Returns
    -------
    threshold_image : (MxNx1) numpy array
        Binarized (0, 255) image as a numpy array.

    """

    # if block_size % 2 != 1:
    #    raise ValueError("block_size must be odd")
    # if type(offset) not in [int, float]:
    #    raise TypeError("offset must be type int or float")
    # if type(gray_image) is not np.ndarray:
    #    raise TypeError("image must be a numpy array")
    # if gray_image.ndim != 2:
    #    raise ValueError("image must be grayscale")
    # if gray_image.dtype is not np.dtype(np.uint8):
    #   raise TypeError("image array must be dtype np.uint8")

    threshold_image = cv2.adaptiveThreshold(gray_image,
                                            255,
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY,
                                            block_size,
                                            offset)

    return threshold_image


def find_contours(threshold_image):

    """
    Returns a list of contours from a binarized thresholded image.

    Parameters
    ----------
    threshold_image : array_like, shape (M, N, 1)
        Binarized threshold image

    Returns
    -------
    contours : list
        List of contours extracted from threshold_image.

    """
    # if len(set([0, 255]) - set(np.unique(threshold_image))) != 0:
    # raise ValueError("image must be binarized to (0, 255)")

    _, contours, _ = cv2.findContours(threshold_image.copy(),
                                      cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    return contours


def fit_polygon(contour, tolerance=0.1):

    """
    Fit a polygonal approximation of a contour up to some tolerance.

    Parameters
    ----------
    contour : array_like
        Contour to fit a polygon to
    tolerance : int or float, default = 0.1
        Tolerance for fitting a polygon as a proportion
        of the perimeter of the contour. This value is used
        to set epsilon, which is the maximum distance between
        the original contour and its polygon approximation.
        Higher values decrease the number of vertices in the polygon.
        Lower values increase the number of vertices in the polygon.
        This parameter affects how many many contours reach the barcode
        matching algorithm, as only polygons with 4 vertices are used.

    Returns
    -------
    polygon : ndarray
        The fitted polygon
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
    Test if a contour is too close to the edge of the image to read.

    Parameters
    ----------
    contour : array_like
        Contour to test
    edge_proximity : int, default = 1
        The distance threshold for ignoring candidate barcodes
        too close to the image edge. Default is 1 pixel.
    x_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the x-axis border. This should correspond to
        frame_width - edge_proximity, but is precalculated for speed.
    y_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the y-axis border. This should correspond to
        frame_height - edge_proximity, but is precalculated for speed.

    Returns
    -------
    edge_proximity_test : bool
        Returns True if the contour is too close to the image edge.

    """
    contour_reshape = np.squeeze(contour)
    contour_x = contour_reshape[:, 0]
    contour_y = contour_reshape[:, 1]
    flat = contour.flatten()

    edge_zero = np.sum(flat <= edge_proximity)
    edge_x = np.sum(contour_x >= x_proximity)
    edge_y = np.sum(contour_y >= y_proximity)

    edge_proximity_test = not (edge_zero == 0 and
                               edge_x == 0 and
                               edge_y == 0)

    return edge_proximity_test


def extract_pixels(gray_image, points, template, max_side, barcode_shape):

    """
    Use affine transform to extract pixel values
    of candidate barcodes from an image.

    Parameters
    ----------
    gray_image : array_like
        Grayscale image from which to extract pixel values
    points : array_like
        An array of coordinates for the corners of the candidate barcode
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
    points = points.reshape((4, 1, 2)).astype(np.float32)
    transform_matrix = cv2.getPerspectiveTransform(points, template)
    pixels = cv2.warpPerspective(gray_image,
                                 transform_matrix,
                                 (max_side, max_side),
                                 flags=(cv2.INTER_NEAREST),
                                 borderValue=255)
    pixels = cv2.resize(pixels, barcode_shape, interpolation=cv2.INTER_LINEAR)
    pixels = np.fliplr(pixels)
    pixels = pixels.reshape((1, -1))

    return pixels


def order_points(polygon):

    """
    Reorder corners of a candidate barcode clockwise from the top-left corner
    (relative to the image).

    Parameters
    ----------
    polygon : array_like, shape (4,1,2)
        A polygon with four coordinates to reorder

    Returns
    -------
    points : ndarray
        The coordinates ordered clockwise from the top-left corner.

    """

    polygon = np.squeeze(polygon)
    points = _order_points(polygon)
    points = polygon.reshape((1, 4, 2))

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

    area_test = (np.sign(area) == area_sign and
                 area_min <= np.abs(area) <= area_max)

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

    # if 4 vertices, sign is correct,
    # value is within range,
    # and polygon is convex...

    quad_test = (polygon.shape[0] == 4 and
                 test_area(polygon_area, area_min, area_max, area_sign) and
                 cv2.isContourConvex(polygon))

    return quad_test


def contour_area(contour):

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


def test_geometry(contour, area_min, area_max,
                  area_sign, edge_proximity, x_proximity,
                  y_proximity, tolerance):

    """
    Test the geometry of a contour.
    Return the corners if it is a candidate barcode.

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
        The distance threshold for ignoring candidate barcodes
        too close to the image edge. Default is 1 pixel.
    x_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the x-axis border. This should correspond to
        frame_width - edge_proximity, but is precalculated for speed.
    y_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the y-axis border. This should correspond to
        frame_height - edge_proximity, but is precalculated for speed.
    tolerance : int or float, default = 0.1
        Tolerance for fitting a polygon as a proportion
        of the perimeter of the contour. This value is used
        to set epsilon, which is the maximum distance between
        the original contour and its polygon approximation.
        Higher values decrease the number of vertices in the polygon.
        Lower values increase the number of vertices in the polygon.
        This parameter affects how many many contours reach the barcode
        matching algorithm, as only polygons with 4 vertices are used.

    Returns
    -------
    geometry_test : bool
        Returns True if the contour is a candidate barcode.
    polygon : ndarray
        The polygon approximation of the contour
    """

    geometry_test = False
    polygon = np.zeros((0, 4, 2))

    if contour.shape[0] >= 4:

        edge_proximity_test = test_edge_proximity(contour,
                                                  edge_proximity,
                                                  x_proximity,
                                                  y_proximity)

        if edge_proximity_test is False:

            area = contour_area(contour)  # calculate the signed area

            # if the sign is correct and value is within range...
            if test_area(area, area_min, area_max, area_sign):

                (polygon, polygon_area, perimeter) = fit_polygon(contour,
                                                                 tolerance)
                peri_area_ratio = np.abs(perimeter / area)

                # if perimeter area ratio is reasonable value...

                if 1 > peri_area_ratio > 0:

                    geometry_test = test_quad(polygon,
                                              polygon_area,
                                              area_min,
                                              area_max,
                                              area_sign)

    return (geometry_test, polygon)


def find_candidate_barcodes(image, contours, barcode_shape,
                            area_min, area_max, area_sign,
                            edge_proximity, x_proximity, y_proximity,
                            tolerance, template, max_side):

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
        The distance threshold for ignoring candidate barcodes
        too close to the image edge. Default is 1 pixel.
    x_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the x-axis border. This should correspond to
        frame_width - edge_proximity, but is precalculated for speed.
    y_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the y-axis border. This should correspond to
        frame_height - edge_proximity, but is precalculated for speed.
    tolerance : int or float, default = 0.1
        Tolerance for fitting a polygon as a proportion
        of the perimeter of the contour. This value is used
        to set epsilon, which is the maximum distance between
        the original contour and its polygon approximation.
        Higher values decrease the number of vertices in the polygon.
        Lower values increase the number of vertices in the polygon.
        This parameter affects how many many contours reach the barcode
        matching algorithm, as only polygons with 4 vertices are used.
    template : array_like
        A template for storing the extracted pixels.
        This should be precalculated using tag_template()
    max_side : int
        The size of the template in pixels

    Returns
    -------
    points_array : ndarray, shape (n_samples, 4, 2)
        Array of coordinates for candidate barcodes.
    pixels_array : ndarray, shape (n_samples, n_pixels)
        Array of flattened pixels for candidate barcodes.
    """

    points_array = np.zeros((0, 4, 2), dtype=np.int32)
    pixels_array = np.zeros((0, barcode_shape[0] * barcode_shape[1]),
                            dtype=np.uint8)

    if len(contours) > 0:  # if contours are found

        # iterate over the list of contours
        for (idx, contour) in enumerate(contours):

            (geometry_test, polygon) = test_geometry(contour,
                                                     area_min,
                                                     area_max,
                                                     area_sign,
                                                     edge_proximity,
                                                     x_proximity,
                                                     y_proximity,
                                                     tolerance)

            if geometry_test is True:

                points = order_points(polygon)
                pixels = extract_pixels(image,
                                        points,
                                        template,
                                        max_side,
                                        barcode_shape)

                points_array = np.append(points_array, points, axis=0)
                pixels_array = np.append(pixels_array, pixels, axis=0)

    # get subpixel coordinates
    shape = points_array.shape
    subpix_shape = (shape[0] * shape[1], 1, shape[2])
    subpix = points_array.reshape(subpix_shape)
    subpix = subpix.astype(np.float32)

    term_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                     500, 0.0001)
    subpix = cv2.cornerSubPix(image, subpix, (5, 5), (-1, -1), term_criteria)
    points_array = subpix.reshape(shape)

    return (points_array, pixels_array)


def match_barcodes(points_array, pixels_array, barcode_nn,
                   id_list, id_index, distance_threshold):

    """
    Match candidate barcodes

    Parameters
    ----------
    points_array : array_like, shape (n_samples, 4, 2)
        Array of coordinates for candidate barcodes
    pixels_array : array_like, shape (n_samples, n_pixels)
        Array of flattened pixels for candidate barcodes
    barcode_nn : NearestNeighbors class
        A NearestNeighbors class from sci-kit learn fitted to
        a list of barcodes. See sklearn.neighbors.NearestNeighbors
        for details
    id_list : array_like
        An array of known identities corresponding
        to the list of barcodes in barcode_nn
    id_index : array_like
        An array of indices corresponding to id_list
    distance_threshold : float
        The maximum distance between a barcode candidate
        and its matched identity


    Returns
    -------
    points_array : ndarray, shape (n_samples, 4, 2)
        Array of coordinates for matched barcodes.
    pixels_array : ndarray, shape (n_samples,
                                   barcode_shape[0] * barcode_shape[1])
        Array of flattened pixels for matched barcodes.
    best_id_list : ndarray
        Array of identities for matched barcodes
    best_id_index : ndarray
        Array of indices corresponding to best_id_list
    distances : ndarray
        Array of distances for matched barcodes
    """

    distances, index = barcode_nn.kneighbors(pixels_array // 255,
                                             n_neighbors=1)
    distances = distances[:, 0]
    index = index[:, 0]

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
    Sort coordinates of matched barcode corners in
    clockwise order from the true top-left corner
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

        tr = points[0]
        tl = points[1]
        bl = points[2]
        br = points[3]

        corners = np.empty_like(points)

        if rotation_test == 3:
            corners[1] = tr
            corners[0] = tl
            corners[3] = bl
            corners[2] = br

        if rotation_test == 0:
            corners[1] = tl
            corners[0] = bl
            corners[3] = br
            corners[2] = tr

        if rotation_test == 1:
            corners[1] = bl
            corners[0] = br
            corners[3] = tr
            corners[2] = tl

        if rotation_test == 2:
            corners[1] = br
            corners[0] = tr
            corners[3] = tl
            corners[2] = bl

        points_array[idx] = corners

    return points_array


def tag_template(max_side):
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
                        dtype=np.float32)

    return template


def test_pixel_variance(points_array, pixels_array, var_thresh=500):
    """
    Test pixel variance of each candidate barcode and
    return only those with variance higher than a set threshold.
    Low variance typically means the candidate barcode is a white
    or black blob and not an actual barcode.

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
        Array of coordinates for candidate barcodes.
    pixels_array : ndarray, shape (n_samples, n_pixels)
        Array of flattened pixels for candidate barcodes.
    """

    variances = np.var(pixels_array, axis=1)
    variance_index = np.where(variances >= var_thresh)
    pixels_array = pixels_array[variance_index]
    points_array = points_array[variance_index]

    return (points_array, pixels_array)


def otsu_threshold(pixels_array):
    """
    Threshold barcode pixels using Otsu's method.
    The algorithm assumes the image contains two classes
    of pixels following a bimodal histogram. It calculates the
    optimum threshold separating the two classes
    so that their combined intra-class variance
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

    for idx, pixels in enumerate(pixels_array):

        ret, pixels = cv2.threshold(pixels, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    shape = (pixels_array.shape[0], barcode_shape[0], barcode_shape[1])
    pixels_array = pixels_array.reshape(shape)
    edge_array = np.ones_like(pixels_array)
    edge_array[:, white_width:-white_width, white_width:-white_width] = 0
    pixels_array[edge_array == 1] = 255
    pixels_array = pixels_array.reshape((pixels_array.shape[0], -1))

    return pixels_array


def preprocess_pixels(points_array, pixels_array, var_thresh,
                      barcode_shape, white_width):
    """
    Preprocess candidate barcode pixels before matching.
    Test pixel variance, binarize the pixel values,
    and whiten the edge pixels for each barcode

    Parameters
    ----------
    pixels_array : ndarray, shape (n_samples, n_pixels)
        Array of flattened pixels for candidate barcodes

    Returns
    -------
    points_array : ndarray, shape (n_samples, 4, 2)
        Array of coordinates for candidate barcodes
    pixels_array : ndarray, shape (n_samples, n_pixels)
        Array of flattened pixels for candidate barcodes
    """

    points_array, pixels_array = test_pixel_variance(points_array,
                                                     pixels_array,
                                                     var_thresh)
    if pixels_array.shape[0] > 0:
        pixels_array = otsu_threshold(pixels_array)
        pixels_array = whiten_edge_pixels(pixels_array,
                                          barcode_shape,
                                          white_width)

    return (points_array, pixels_array)


def process_frame(frame, channel, resize,
                  block_size, offset, barcode_shape,
                  white_width, area_min, area_max,
                  x_proximity, y_proximity, tolerance,
                  template, max_side, var_thresh,
                  barcode_nn, id_list, id_index,
                  distance_threshold, clahe, otsu, dilate):
    """
    Process a single frame to track barcodes.

    Parameters
    ----------
    frame : array_like, shape (MxNx3)
        3-channel color image
    channel : {0, 1, 2, None}, default = None
        The color channel to use for producing the grayscale image.
    block_size : int, default = 1001
        Odd value integer. Size of the local neighborhood
        for adaptive thresholding.
    offset : default = 2
        Constant subtracted from the mean.
        Normally, it is positive but may be zero or negative as well.
        The threshold value is the mean of the block_size x block_size
        neighborhood minus offset.
    barcode_shape : tuple
        The shape of the barcode with white border
    white_width : int
        The bit width of the white border around the barcode
    area_min : float
        Minimum area
    area_max : float
        Maximum area
    edge_proximity : int, default = 1
        The distance threshold for ignoring candidate barcodes
        too close to the image edge. Default is 1 pixel.
    x_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the x-axis border. This should correspond to
        frame_width - edge_proximity, but is precalculated for speed.
    y_proximity : int
        The distance threshold for ignoring candidate barcodes
        near the y-axis border. This should correspond to
        frame_height - edge_proximity, but is precalculated for speed.
    tolerance : int or float, default = 0.1
        Tolerance for fitting a polygon as a proportion
        of the perimeter of the contour. This value is used
        to set epsilon, which is the maximum distance between
        the original contour and its polygon approximation.
        Higher values decrease the number of vertices in the polygon.
        Lower values increase the number of vertices in the polygon.
        This parameter affects how many many contours reach the barcode
        matching algorithm, as only polygons with 4 vertices are used.
    template : array_like
        A template for storing the extracted pixels.
        This should be precalculated using tag_template()
    max_side : int
        The size of the template in pixels
    var_thresh : float, (default = 500)
        Minimum variance threshold
    barcode_nn : NearestNeighbors class
        A NearestNeighbors class from sci-kit learn fitted to
        a list of barcodes. See sklearn.neighbors.NearestNeighbors
        for details
    id_list : array_like
        An array of known identities corresponding
        to the list of barcodes in barcode_nn
    id_index : array_like
        An array of indices corresponding to id_list
    distance_threshold : float
        The maximum distance between a barcode candidate
        and its matched identity

    Returns
    -------
    fetch_dict : dict

        A dictionary containing the following objects:

        "gray" : ndarray, shape (MxNx1)
            The grayscale image
        "thresh" : ndarray, shape (MxNx1)
            The threshold image
        "corners" : ndarray, shape (n_samples, 4, 2)
            Array of coordinates for barcodes
        "pixels" : ndarray, shape (n_samples, n_pixels)
            Array of flattened pixels for barcodes
        "identity" : ndarray, shape (n_samples)
            Array of identities that best match each barcode
        "distance" : ndarray, shape (n_samples)
            Array of Hamming distances between each barcode
            and the closest match
    """

    best_id_list = np.zeros((0, 1))
    distances = np.zeros((0, 1))

    gray = grayscale(frame, channel)
    if clahe:
        clahe = cv2.createCLAHE(0.05, (clahe, clahe))
        gray = clahe.apply(gray)
    if resize > 1:
        gray = cv2.resize(gray, (0, 0), None, resize, resize,
                          interpolation=cv2.INTER_CUBIC)
    if otsu:
        ret, thresh = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh = adaptive_threshold(gray, block_size=block_size, offset=offset)

    if dilate:
        thresh = cv2.dilate(thresh, np.ones((3, 3)))

    contours = find_contours(thresh)

    (points_array,
     pixels_array) = find_candidate_barcodes(image=gray,
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
        (points_array,
         pixels_array) = preprocess_pixels(points_array=points_array,
                                           pixels_array=pixels_array,
                                           var_thresh=var_thresh,
                                           barcode_shape=barcode_shape,
                                           white_width=white_width
                                           )

    if pixels_array.shape[0] > 0:
        (points_array,
         pixels_array,
         best_id_list,
         best_id_index,
         distances) = match_barcodes(points_array=points_array,
                                     pixels_array=pixels_array,
                                     barcode_nn=barcode_nn,
                                     id_list=id_list,
                                     id_index=id_index,
                                     distance_threshold=distance_threshold
                                     )
    if points_array.shape[0] > 0:
        points_array = sort_corners(points_array, best_id_index)

    fetch_dict = {"gray": gray,
                  "thresh": thresh,
                  "corners": points_array,
                  #"pixels": pixels_array,
                  "identity": best_id_list,
                  "distance": distances
                  }

    return fetch_dict
