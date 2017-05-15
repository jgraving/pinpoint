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
import utils as utils
from TagDictionary import TagDictionary

from sklearn.neighbors import NearestNeighbors

from types import BooleanType, IntType, StringType, FloatType, NoneType, TupleType
import warnings

cv2.setNumThreads(-1)

__all__ = ['Tracker']


class Tracker(TagDictionary, VideoReader):
    
    def __init__(self, source, block_size=1001, offset=80, area_range=(10,10000), tolerance=0.1, distance_threshold=8, var_thresh=500, channel='green'):
        
        VideoReader.__init__(self, source)
        TagDictionary.__init__(self)
        self.area_min = area_range[0]
        self.area_max = area_range[1]
        self.tolerance = tolerance
        self.distance_threshold = distance_threshold
        self.block_size = block_size
        self.offset = offset
        self.channel = channel
        self.var_thresh = var_thresh
        self.frame_width = self.width()
        self.frame_height = self.height()
        self.x_proximity = self.frame_width-1
        self.y_proximity = self.frame_height-1

        
    def track(self, batch_size=8):
        
        self.barcode_shape = self.white_shape
        max_side=self.barcode_shape[0]
        template = get_tag_template(max_side)
        
        self.barcode_nn = NearestNeighbors(metric='cityblock', algorithm='brute')
        self.barcode_nn.fit(self.barcode_list)
        
        #while not self.finished():
        frames = self.read_batch(20)
        for idx,frame in enumerate(frames):
            thresh, points_array, pixels_array, best_id_list, distances = process_frame(frame=frame,
                                                                                       channel=self.channel,
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
                        
        
        return (thresh, pixels_array, points_array, best_id_list, distances)
        
        


class Tracker(TagDictionary):
	
	"""Initializes a Tracker.

	Parameters
	----------
	source : int or str
		The source for tracking. 

	Returns
	-------
	self : class
		the Tracker class 
	
	Notes
	-------
	The source can be an integer value for a webcam device index or a filepath string to a video file (e.g. "/path/to/file/video.mp4")

	"""

	def __init__(self, source):
		
		self.source = source

		if type(self.source) not in [IntType, StringType]:
			raise TypeError("source must be type int or str")
		
		if type(self.source) is StringType:

			if os.path.splitext(self.source)[1] in [".jpg",".JPG",".png",".PNG",".jpeg",".JPEG",".tiff",".TIFF",".gif",".GIF",".bmp",".BMP"]: 
				self.source_type = "images"
				raise IOError("Images not yet supported")
			else:
				self.source_type = "video"

		elif type(self.source) is IntType:
			self.source_type = "camera"

		self.cap = cv2.VideoCapture(self.source)

		assert self.cap.isOpened(), "Video source failed to open. \nCheck that your camera is connected or that the video file exists."
		
		ret, frame = self.cap.read()
		assert ret, "Video source opened but failed to read any images. \nCheck that the camera is working or the video is a valid file"
		
		self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

		if self.source_type == "video":
			self.fps = self.cap.get(cv2.CAP_PROP_FPS)
		else:
			self.fps = 29.97
		
		self.roi_pt1 = (0,0)
		self.roi_pt2 = (self.frame_width-1,self.frame_height-1)

		self.area_min = 1
		self.area_max = 10000

		self.blur_size = 1
		self.blur_sigma = 0

		self.savefile = None

		self.edge_proximity = 1

		self.channel = None

		self.tolerance = 0.1
		
		self.cap.release()

	def show_video(self, resize = 1):

		"""Shows video source. Press `esc` to exit

			Parameters
			----------
			resize : float, default = 1
				resize factor for showing the video source. 1 = original size
			video_output : str, default = None
				Saves the stream to a file as .mp4, i.e. "/path/to/video.mp4".
				If None, no video is saved.
		"""

		if type(resize) not in [IntType, FloatType]:
			raise TypeError("resize must be type int or float")
		if resize <= 0:
			raise ValueError("resize must be positive, non-zero value")
		
		print("Showing video source. Press `esc` to exit.")

		cap = cv2.VideoCapture(self.source)

		video_title = "Video (press `esc` to exit)."

		cv2.startWindowThread()
		EXIT = False
		while (cap.isOpened() and EXIT == False):
			ret,frame = cap.read()
			if ret:
				if resize != 1.:
					frame = cv2.resize(frame, (0,0), None, fx = resize, fy = resize) 

				cv2.imshow(video_title, frame)

			elif cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
				EXIT = True

			if cv2.waitKey(1) & 0xff == 27:
				EXIT = True

		cap.release()
		cv2.destroyAllWindows()

		for i in range(10):
			cv2.waitKey(1)

		print("Succesfully exited.")

	def show_image(self, frame_number = None, figsize = (10,10), image_output = None):

		"""Capture and show a still image from the video source.
			Parameters
			----------
			frame_number : int, default = None
				Frame number to show from the video. If None, the first frame is used.
			figsize : tuple of int, default = (10,10)
				Figure size for showing the image with matplotlib
			image_output : str, default = None
				Saves the image to a file. If None, no image is saved.
		"""

		if type(image_output) not in [NoneType, StringType]:
			raise TypeError("image_output must be type str or None")
		if image_output is not None:
			if os.path.splitext(image_output)[1] not in [".jpg",".png",".jpeg",".tiff"]:
				raise IOError("Output file must be .jpg, .jpeg, .png, or .tiff")
		if frame_number is not None and self.source_type == 'camera':
			raise IOError("Cannot specify frame_number with camera source")
		if type(frame_number) not in [NoneType, IntType]:
			raise TypeError("frame_number must be type int or None")
		
		
		cap = cv2.VideoCapture(self.source)
		cv2.waitKey(100)

		if frame_number != None:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

		ret, image = cap.read()
		cap.release()
		plt.figure(figsize = figsize)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		plt.imshow(image)
		plt.title(str(image.shape[1]) +"x"+ str(image.shape[0]))
		plt.show()

		if image_output != None:
			cv2.imwrite(image_output, still)

	def set_roi(self, pt1, pt2, imshow = True, figsize = (20,6), frame_number = None):
		"""Set the region of interest for the tracker.

			Parameters
			----------
			pt1 : tuple of int
				top left point (x,y) of the region of interest
			pt2 : tuple of int
				bottom right point (x,y) of the region of interest
			imshow : bool, default = True
				Shows a image with the area of interest drawn over top
			figsize : tuple of int, default = (20,6)
				The size of the imshow image
			frame_number : int, default = None
				Frame number to show from the video 
		"""

		if not (type(pt1) is TupleType and type(pt2) is TupleType):
			raise TypeError("pt1 and pt2 must be type tuple")
		if not (len(pt1) == 2 and len(pt2) == 2):
			raise ValueError("pt1 and pt2 must be length 2")
		if not (all([isinstance(pt, IntType) for pt in pt1]) and all([isinstance(pt, IntType) for pt in pt2])): 
			raise TypeError("pt1 and pt2 must contain only type int")
		if type(frame_number) not in [NoneType, IntType]:
			raise TypeError("frame_number must be type int or None")
		if not (frame_number is None and self.source_type is StringType):
				raise ValueError("Cannot specify frame_number with camera source")

		self.roi_pt1 = pt1
		self.roi_pt2 = pt2

		cap = cv2.VideoCapture(self.source)
		cv2.waitKey(100)

		if frame_number != None:
			cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

		ret, still = cap.read()
		cap.release()

		still = cv2.cvtColor(still, cv2.COLOR_BGR2RGB)
		cropped = utils.crop(still, self.roi_pt1, self.roi_pt2)

		self.frame_height = cropped.shape[0]
		self.frame_width = cropped.shape[1]

		if imshow == True:
			fig, (ax1,ax2) = plt.subplots(1, 2, figsize = figsize)
			roi = cv2.rectangle(still.copy(), pt1, pt2, (0,255,0), 3)
			ax1.imshow(roi)
			ax1.set_title("Region of Interest")

			ax2.imshow(cropped)
			ax2.set_title(str(self.frame_width) +"x"+ str(self.frame_height))
			plt.tight_layout()
			plt.show()

	def set_area(self, area_min = 1, area_max = 10000):
		"""Set the area range for the tracker.

			Parameters
			----------
			area_min : int, default = 10
				Minimum area a contour must have to be tracked
			area_max : int, default = 10000
				Maximum area a contour can have to be tracked
		"""

		if type(area_min) is not IntType:
			raise TypeError("area_min must be type int")
		if type(area_max) is not IntType:
			raise TypeError("area_max must be type int")

		self.area_min = area_min
		self.area_max = area_max

	def set_threshold(self, block_size = 1001, offset = 0):
		"""Set the adaptive threshold parameters for the tracker.

			Parameters
			----------
			block_size : int, default = 1001
				Odd value integer. Size of the local neighborhood for adaptive thresholding.
			offset : default = 0
				Constant subtracted from the mean. It may be positive, zero, or negative.
			Notes
			----------	
			The threshold value T(x,y) is calculated as the mean of the block_size x block_size neighborhood minus offset.
		"""

		if block_size % 2 != 1:
			raise ValueError("block_size must be an odd value (block_size % 2 == 1)")
		if type(offset) not in [int, float]:
			raise TypeError("offset must be type int or float")

		self.block_size = block_size
		self.offset = offset

	def set_color_channel(self, channel = None):

		"""Set the color channel for generating a grayscale image from a color image.

			Parameters
			----------
			channel : str, default = None
				The color channel to use for producing the grayscale image.

			Notes
			----------
			None uses the default linear transformation from OpenCV: 
			Y = 0.299R + 0.587G + 0.114B
			Channels 'blue', 'green', and 'red' use the respective color channel as the grayscale image. 
			Under white, broad-spectrum ambient lighting, 'green' typically provides the lowest noise.
		"""

		if type(channel) not in [StringType, NoneType]:
			raise TypeError("channel must be type str or None")
		if type(channel) is StringType and not (channel.startswith('b') or channel.startswith('g') or channel.startswith('r')):
			raise ValueError("channel value must be 'blue', 'green', 'red', or None")

		self.channel = channel

	def set_tolerance(self, tolerance = 0.1):
		"""Set the polygon tolerance for the tracker.

			Parameters
			----------
			tolerance : int or float, default = 0.1
				Tolerance for fitting a polygon as a proportion of the perimeter of the contour. 

			Notes
			----------
			This value is used to set epsilon, which is the maximum distance between the original contour and its polygon approximation. 
			Higher values decrease the number of vertices in the polygon.
			Lower values increase the number of vertices in the polygon.
			This parameter affects how many many contours reach the barcode matching algorithm, as only polygons with 4 vertices are used.
		"""

		if type(tolerance) not in [IntType, FloatType]:
			raise TypeError("tolerance must be type int or float")
		if tolerance >= 1:
			warnings.warn("A tolerance value <1.0 is recommended")

		self.tolerance = tolerance

	def track(self, data_output = None, video_output = None):

		"""Starts the tracker with options to save the tracking data and record video to a file.

			Parameters
			----------
			data_output : str, default = None
				File path to the text file where the data will be stored as comma separated values.
				Must be file extension .txt or .csv
				If None, no data will be stored.
			video_output. : str, default = None
				File path to the video file where the output video will be stored as MPEG-4.
				Must be file extension .mp4 or .MP4
				If None, no video will be recorded.

		"""

		if type(data_output) not in [StringType, NoneType]:
			raise TypeError("data_output must be type str or None")
		if data_output is not None and os.path.splitext(data_output)[1] not in [".csv",".txt"]:
			raise IOError("Data file must be .csv, or .txt")

		if type(video_output) not in [StringType, NoneType]:
			raise TypeError("video_output must be type str or None")
		if video_output is not None and os.path.splitext(video_output)[1] not in [".MP4",".mp4"]:
			raise IOError("Video output file must be .mp4")

		timestamp = dt.datetime.now()
		frame_number = 0

		if data_output is not None:

			directory = os.path.dirname(data_output)
			if not os.path.exists(directory):
				os.makedirs(directory)

			self.savefile = open(data_output,"w+")

			if self.source_type == 'video':
				write_str = "msec" + "," + "frame_number" + "," + "id" + "," + "correlation" + "," + "x" + "," + "y" + "," + "orientation" + "\n"
			elif self.source_type == 'camera':
				write_str = "msec" + "," + "frame_number" + "," + "id" + "," + "correlation" + "," + "x" + "," + "y" + "," + "orientation" + "\n"
			
			self.savefile.write(write_str)

		self.cap = cv2.VideoCapture(self.source)

		self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

		if video_output is not None:
			fourcc = cv2.VideoWriter_fourcc("A","V","C","1")
			out = cv2.VideoWriter(video_output, fourcc, self.fps, (self.frame_width,self.frame_height), True)

		cv2.namedWindow("Tracker")
		cv2.startWindowThread()

		self.x_proximity = self.frame_width - self.edge_proximity - 1
		self.y_proximity = self.frame_height - self.edge_proximity - 1

		max_side = 7

		dst = utils.get_warp_dst(max_side)

		while cap.isOpened():

			ret, frame = cap.read()

			if self.source_type == 'camera':
				frame_number += 1
			if self.source_type == 'video':
				frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)

			if ret:
				cropped = utils.crop(frame, self.pt1, self.pt2)
				gray = utils.get_grayscale(cropped, channel = self.channel)
				thresh = utils.get_threshold(gray, block_size = self.block_size, offset = self.offset)
				contours = utils.get_contours(thresh)

				(points_array, pixels_array) = utils.get_candidate_barcodes(image = gray,
																			contours = contours,
																			barcode_size = self.barcode_size,
																			area_min = self.area_min,
																			area_max = self.area_max,
																			area_sign = self.area_sign,
																			edge_proximity = self.edge_proximity,
																			x_proximity = self.x_proximity,
																			y_proximity = self.y_proximity,
																			tolerance = self.tolerance
																			)

				if points_array is not None and pixels_array is not None:
					

					

				if data_output != None:
					if self.source_type == 'video':
						write_str = str(cap.get(cv2.CAP_PROP_POS_MSEC)) + "," + str(frame_number) + "," + str(mcx) + "," + str(mcy) + "\n"
					#elif self.source_type == 'camera':
					#	write_str = str(time) + "," + str(frame_number) + "," + str(mcx) + "," + str(mcy) + "\n"
					self.savefile.write(write_str)
				
				cv2.imshow("Tracker",frame)

				if video_output != None:
					out.write(frame)
								
			k = cv2.waitKey(int(self.delay*1000)) & 0xff
			if k == 27:
				break
			elif frame_number == total_frames:
				break
		 
		cap.release()
		cv2.destroyWindow("Tracker")

		if data_output != None:
			self.savefile.close()     
		if video_output != None:
			out.release()

		for i in range(10):
			cv2.waitKey(1)