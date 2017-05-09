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
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from types import BooleanType, IntType, StringType, FloatType, NoneType, TupleType
from .utils import rotate_tag90

def pairwise_distance_check(X, Y=None, distance=5, metric='cityblock', n_jobs=1):
	""" Check the pairwise distances between two arrays.
	
	Parameters
	----------
	X : array
		Array of flattened barcodes.
	Y : array
		Array of flattened barcodes.
	distance : float
		Minimum distance between all barcodes in X and Y.
		
	Returns
	-------
	test : bool
		Returns True if all elements are greater than the minimum distance.
	
	"""
	D = pairwise_distances(X, Y, metric=metric, n_jobs=n_jobs)
	D[D < distance] = 0
	D[D >= distance] = 1

	if type(Y) == np.ndarray:
		if D.sum() == D.size:
			test = True
		else:
			test = False
	
	elif type(Y) == NoneType:
		if D.sum() == D.size - D.shape[0]:
			test = True
		else:
			test = False
	
	return test

class TagDictionary:

	"""A class for generating, saving, loading, and printing 2-D barcode tags.

		Parameters
		----------
		tag_shape : tuple of int, default = (5,5)
			Shape of the barcodes to generate.
		distance : float, default = 7
			Minimum distance between barcodes
		white_width : int, default = 1
			Width of white border in bits.
		black_width : int, default = 1
			Width of black border in bits.
		niter : int, default = 99999
			Number of iterations to try.
		metric : str, default = 'cityblock'
			The distance metric to use for generating the barcodes.
		"""
    
	def __init__(self, tag_shape = (5,5), distance = 7, white_width = 1, black_width = 1, metric='cityblock'):

		self.tag_shape = tag_shape
		self.distance = distance
		self.metric = metric
		self.white_width = white_width
		self.black_width = black_width
		self.white_shape = tuple([x+(self.white_width*2) for x in self.tag_shape])
		self.black_shape = tuple([x+(self.white_width*2)+(self.black_width*2) for x in self.tag_shape])

		self.first_tag = True # start with first tag
		self.tag_len = self.tag_shape[0]*self.tag_shape[1] # get length of flattened tag

		self.master_list = None
		self.loaded = False
		self.saved = False
	
	def generate_dict(self, niter = 99999, verbose = False, reset_seed = True, n_jobs=1):
		"""Start generating barcode tags. Speed depends on hardware, tag size, and the number of iterations. This may take a few minutes."""
		
		self.niter = niter
		if reset_seed == True:
			new_seed = np.random.randint(0, 32768)
			np.random.seed(new_seed)
			self.random_seed = new_seed

		if verbose:
			print "Generating tags. This may take awhile..."

		for idx in np.arange(0, self.niter + 1): # generate some tags

			if verbose and idx % 10000 == 0 and idx > 0 and type(self.master_list) != NoneType:
				print "Iteration: " + str(idx) + "/" + str(self.niter)
				print "Tags found: ", len(self.master_list)//4

			if verbose and idx == self.niter and type(self.master_list) != NoneType:
				print "Iteration: " + str(idx) + "/" + str(self.niter)
				print "Tags found: ", len(self.master_list)//4

			new_tag = np.random.randint(0,2, size=(self.tag_len)).astype(np.uint8) # randomly generate a tag

			# get tag rotations
			tag_90 = rotate_tag90(new_tag, self.tag_shape, 1) 
			tag_180 = rotate_tag90(new_tag, self.tag_shape, 2)
			tag_270 = rotate_tag90(new_tag, self.tag_shape, 3)
			tag_list = np.asarray([new_tag, tag_90, tag_180, tag_270]).astype(np.uint8)

			# check distance between tag and rotations
			test = pairwise_distance_check(tag_list, distance=self.distance, metric=self.metric, n_jobs=n_jobs)

			if test: # if tag and all rotations are different enough from each other...

				if self.first_tag == True:
					self.master_list = tag_list
					self.first_tag = False # done with first tag

				elif self.first_tag == False:
					
					# check distance between all tag rotations and master list of tags
					test = pairwise_distance_check(X=tag_list, Y=self.master_list, distance=self.distance, metric=self.metric, n_jobs=n_jobs)
					
					if test:  # if all tag rotations are different enough from master list...
						for tag in tag_list:
							tag = np.asarray([tag]).astype(np.uint8)
							self.master_list = np.append(self.master_list, tag, axis=0) 

		self.ntags = self.master_list.shape[0]//4

		id_list = []

		for idx in range(self.ntags):
			ID = [idx,idx,idx,idx]
			id_list.append(ID)

		id_list = np.array(id_list) + 1
		self.id_list = id_list.flatten()
	
		if verbose:
			print "Done!"

		return self
		
	def save_dict(self, filename = "master_list.pkl"):

		"""Save configuration as ``.pkl`` file.

		Parameters
		----------
		filename : str, default = "master_list.pkl"
			Path to save file, must be '.pkl' extension

		Returns
		-------
		saved : bool
			Saved successfully.
	"""
		config = {  "master_list" : self.master_list,
					"id_list" : self.id_list, 
					"tag_shape" : self.tag_shape, 
					"distance" : self.distance, 
					"white_width" : self.white_width, 
					"black_width" : self.black_width, 
				 }

		output = open(filename, 'wb')

		try:
			pickle.dump(config, output)
		except:
			raise IOError("File must be '.pkl' extension")

		output.close()

		self.saved = True

		return self.saved

	def load_dict(self, filename = "master_list.pkl"):

		"""Load configuration from a ``.pkl`` file.

		Parameters
		----------
		filename : str, default = "master_list.pkl"
			Path to load file, must be '.pkl' extension

		Returns
		-------
		loaded : bool
			Loaded successfully.
		"""

		# Open and load file
		pkl_file = open(filename, 'rb')

		try:
			config = pickle.load(pkl_file)
		except:
			raise IOError("File must be '.pkl' extension")

		pkl_file.close()

		# Load new configuration
		self.master_list = config["master_list"]
		self.id_list = config["id_list"]
		self.tag_shape = config["tag_shape"]
		self.distance = config["distance"]
		self.white_width = config["white_width"]
		self.black_width = config["black_width"]
		self.white_shape = tuple([x+(self.white_width*2) for x in self.tag_shape])
		self.black_shape = tuple([x+(self.white_width*2)+(self.black_width*2) for x in self.tag_shape])
		self.ntags = self.master_list.shape[0]//4
		self.tag_len = self.tag_shape[0]*self.tag_shape[1]

		self.loaded = True

		return self.loaded


	def print_tags(self, file, ntags = 200, page_size = (8.26, 11.69), ncols = 20, id_fontsize = 5, arrow_fontsize = 10, id_digits = 4, show = True):

		"""Print tags as image file or PDF. Default settngs are for ~6-mm wide tags.

		Parameters
		----------
		file : str
			Location for saving the barcode images, must be `.pdf` or image (`.png`, `.jpg`, etc.) file extension.
		ntags : int, default = 200
			Number of tags per page.
		page_size : tuple of float, default = (8.26, 11.69)
			Size of the printed page, default is A4. 
		ncols : int, default = 20
			Number of columns.
		id_fontsize : int, default = 5
			Font size for ID number.
		arrow_fontsize : int, default = 10
			Font size for arrow.
		id_digits : int, default = 5
			Number of digits for ID number printed below barcode (zero pads the left side of the number).
		show : bool
			Show the figure using plt.show()

		"""

		self.fig = plt.figure(figsize = page_size)
		plot = 1
		for idx, tag in enumerate(self.master_list):
			if (idx+1) % 4 == 0 and plot <= ntags:

				tag = add_border(tag, self.tag_shape, self.white_width, self.black_width)
				tag = tag.reshape(self.black_shape)

				ax = plt.subplot(ntags//ncols, ncols, plot)
				tag_number = str((idx+1)/4)

				if len(tag_number) < id_digits:
					tag_number = "0"*(id_digits-len(tag_number)) + tag_number
				
				ax.set_title(u"\u2191", fontsize = arrow_fontsize, family = "Arial", weight = "heavy")
				ax.set_xlabel(tag_number, fontsize = id_fontsize, family = "Arial", weight = 'light', color = 'white', backgroundcolor = 'black', bbox = {'fc': 'black', 'ec': 'none'})
				ax.set_aspect(1)
				ax.xaxis.set_ticks_position('none')
				ax.yaxis.set_ticks_position('none')
				ax.get_xaxis().set_ticks([])
				ax.get_yaxis().set_ticks([])
				ax.xaxis.set_label_coords(0.5, -0.1)

				ax.imshow(tag, cmap='gray', interpolation = 'nearest', zorder = 200)

				plot += 1

		plt.savefig(file, dpi=600, interpolation = 'none')

		if show == True:
			plt.show()

		return True


