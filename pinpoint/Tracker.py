#! /usr/bin/env python

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

from __future__ import division, print_function

import cv2
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os.path

from .TagDictionary import TagDictionary
from .CameraCalibration import CameraCalibration

from sklearn.neighbors import NearestNeighbors

import h5py

from .utils import process_frame, get_tag_template

from types import BooleanType, IntType, StringType, FloatType, NoneType, TupleType
import warnings

cv2.setNumThreads(-1)

__all__ = ['Tracker']

class VideoReader:
	
	def __init__(self, path):
		
		self.stream = cv2.VideoCapture(path)
		self._total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
		self._fps = self.stream.get(cv2.CAP_PROP_FPS)
		self._codec = self.stream.get(cv2.CAP_PROP_FOURCC)
		self._height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self._width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
		self._finished = False

	def read(self):
		ret, frame = self.stream.read()
		if ret:
			return frame
		else:
			self._finished = True
	
	def read_batch(self, batch_size=8):
		frames = []
		for idx in range(batch_size):
			ret, frame = self.stream.read()
			if ret:
				frames.append(frame)
			else:
				self._finished = True
		return frames
	
	def close(self):
		self.stream.release()
		return not self.stream.isOpened()
	
	def current_frame(self):
		return int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
	
	def current_time(self):
		return self.stream.get(cv2.CAP_PROP_POS_MSEC)
	
	def percent_finished(self):
		return self.stream.get(cv2.CAP_PROP_POS_AVI_RATIO) * 100
	
	def fps(self):
		return self._fps
	
	def codec(self):
		return self._codec
	
	def height(self):
		return self._height
	
	def width(self):
		return self._width
	
	def total_frames(self):
		return self._total_frames
	
	def finished(self):
		return self._finished

class Tracker(TagDictionary, VideoReader, CameraCalibration):
	"""
	
	Tracker class for processing videos to track barcodes. 

	Parameters
	----------
	block_size : int, default = 1001
		Odd value integer. Size of the local neighborhood for adaptive thresholding.
	offset : default = 2
		Constant subtracted from the mean for adaptive thresholding. Normally, it is positive but 
		may be zero or negative as well. The threshold value is calculated as the mean of the 
		block_size x block_size neighborhood *minus* the offset.
	area_range : tuple, default (10,10000)
		Area range in pixels for potential barcodes. If the minimum value is too low this
		can lead to false positives.
	tolerance : int or float, default = 0.1
		This parameter affects how many many contours reach the barcode matching algorithm, 
		as only polygons with 4 vertices are used. This is the toleracne for fitting a polygon as a 
		proportion of the perimeter of the contour. This value is used to set epsilon, which is the 
		maximum distance between the original contour and its polygon approximation. 
		Higher values decrease the number of vertices in the polygon.
		Lower values increase the number of vertices in the polygon. 
	distance_threshold : int, default = 8
		The maximum Hamming distance between a barcode candidate and its matched identity.
		Set this to some high value to save all candidates.
	var_thresh : float, (default = 500)
		Minimum variance threshold. Candidate barcodes with low variance are likely white or black blobs.
	channel : {'blue', 'green', 'red', 'none', None}, default = None
		The color channel to use for producing the grayscale image.
	resize : float, default=1.0
		The scalar for resizing images. In most cases, increasing the size of the image 
		can improve edge detection which leads to better barcode reconstruction 
		at the expense of computation time. The recommended setting is
		some value between 1.0 and 2.0
	
	Returns
	-------
	self : class
		Tracker class instance
	
	"""

	def __init__(self, source, block_size=1001, offset=80, area_range=(10,10000), tolerance=0.1, distance_threshold=8, var_thresh=500, channel='green', resize=1.0):
		
		VideoReader.__init__(self, source)
		TagDictionary.__init__(self)
		CameraCalibration.__init__(self)
		self.area_min = area_range[0]*(resize**2)
		self.area_max = area_range[1]*(resize**2)
		self.tolerance = tolerance
		self.distance_threshold = distance_threshold
		self.block_size = block_size
		self.offset = offset
		self.channel = channel
		self.var_thresh = var_thresh
		self.resize = resize
		self.frame_width = self.width()
		self.frame_height = self.height()
		self.x_proximity = (self.frame_width*self.resize)-1
		self.y_proximity = (self.frame_height*self.resize)-1

		
	def track(self, filename='output.h5', batch_size=8):
		
		"""
		Process frames to track barcodes. Saves data to hdf5 file. See `Notes` for details.

		Parameters
		----------
		filename : str, default = 'output.h5'
			The file to save data to. 
		batch_size : int, default is 8
			The number of frames to process in each batch.
			
		Notes
		-----
		The tracker outputs data as an hdf5 file with the following structure:
			--filename
			----/data
			------/frame_idx
			------/corners
			------/identity
			------/distances
		frame_idx : ndarray, dtype=int, shape (n_samples, 1)
			the frame index number from the video for each sample 
		corners : ndarray, dtype=float, shape (n_samples, 4, 2)
			The corners for each barcode for all frames
		identity : ndarray, dtype=int, shape (n_samples, 1)
			The nearest identity for each sample
		distances : ndarray, dtype=int, shape (n_samples, 1)
			The Hamming distance for each sample to the 
			nearest neighbor from the barcode dictionary
			
			
		Returns
		-------
		gray : ndarray, (height, width, 1)
			Single-channel grayscale image from the most recent frame
		thresh : ndarray, (height, width, 1)
			Single-channel threshold image from the most recent frame
		points_array : ndarray, shape (n_samples, 4, 2)
			Array of coordinates for barcodes from the most recent frame
		pixels_array : ndarray, shape (n_samples, n_pixels)
			Array of flattened pixels for barcodes from the most recent frame
		best_id_list : ndarray, dtype=int, shape (n_samples, 1)
			The nearest identity for each sample
		distances : ndarray, dtype=int, shape (n_samples, 1)
			The Hamming distance for each sample to the 
			nearest neighbor from the barcode dictionary
		"""
		
		self.h5file = h5py.File(filename, 'w')
		dgroup = self.h5file.create_group('data')
		
		frame_idx_dset = dgroup.create_dataset('frame_idx', shape=(0,), dtype=np.int64, maxshape=(None,))
		corners_dset = dgroup.create_dataset('corners', shape=(0,4,2), dtype=np.float64, maxshape=(None,4,2))
		identity_dset = dgroup.create_dataset('identity', shape=(0,), dtype=np.int32, maxshape=(None,))
		distances_dset = dgroup.create_dataset('distances', shape=(0,), dtype=np.int32, maxshape=(None,))
		
		dset_list = [frame_idx_dset, corners_dset, identity_dset, distances_dset]
		
		self.barcode_shape = self.white_shape
		max_side=100
		template = get_tag_template(max_side)
		
		self.barcode_nn = NearestNeighbors(metric='cityblock', algorithm='ball_tree')
		self.barcode_nn.fit(self.barcode_list)
		#dists = []
		idx = 0
		try:
			while not self.finished():
			#for idx in range(1):
				frames = self.read_batch(batch_size)
				for frame in frames:
					gray, thresh, points_array, pixels_array, best_id_list, distances = process_frame(frame=frame,
																							   channel=self.channel,
																							   resize=self.resize,
																							   block_size=self.block_size,
																							   offset = self.offset,
																							   barcode_shape=self.barcode_shape,
																							   white_width=self.white_width,
																							   area_min=self.area_min,
																							   area_max=self.area_max,
																							   x_proximity=self.x_proximity,
																							   y_proximity=self.y_proximity,
																							   tolerance=self.tolerance,
																							   max_side=max_side,
																							   template=template,
																							   var_thresh=self.var_thresh,
																							   barcode_nn=self.barcode_nn,
																							   id_list=self.id_list,
																							   id_index=self.id_index,
																							   distance_threshold=self.distance_threshold,
																							  )
					points_array = points_array / self.resize
					frame_idx = np.repeat(idx, points_array.shape[0])
					idx += 1
					data_list = [frame_idx, points_array, best_id_list, distances]
					for (dset, data) in zip(dset_list, data_list):

						current_shape = list(dset.shape)
						current_size = current_shape[0]

						new_shape = current_shape
						new_shape[0] = new_shape[0]+data.shape[0]
						new_size = new_shape[0]

						dset.resize(tuple(new_shape))
						dset[current_size:new_size] = data

		except KeyboardInterrupt:
			self.h5file.close()
		
		self.h5file.close()

		return (gray, thresh, points_array, pixels_array, best_id_list, distances)