""" Class containing methods for generating, saving, loading, and printing 2-D barcode tags """

import numpy as np
import matplotlib.pyplot as plt
import pickle
import numpy as np
import utils


class TagDictionary:

	"""Class for generating, saving, loading, and printing 2-D barcode tags.

		Parameters
		----------
		tag_shape : tuple of int, default = (5,5)
			Shape of the barcodes to generate.
		ndiffs : int, default = 7
			Minimum number of bitwise differences (Hamming distance).
		white_width : int, default = 1
			Width of white border in bits.
		black_width : int, default = 1
			Width of black border in bits.
		niter : int, default = 99999
			Number of iterations to try.

		"""
	def __init__(self, tag_shape = (5,5), ndiffs = 7, white_width = 1, black_width = 1):

		self.tag_shape = tag_shape
		self.ndiffs = ndiffs
		self.white_width = white_width
		self.black_width = black_width

		self.first_tag = True # start with first tag
		self.tag_len = self.tag_shape[0]*self.tag_shape[1] # get length of flattened tag

		self.master_list = None
		self.loaded = False
		self.saved = False
	
	def generate_dict(self, niter = 99999, verbose = False, reset_seed = True):
		"""Start generating barcode tags. Speed depends on hardware, tag size, and the number of iterations. This may take a few minutes."""
		
		self.niter = niter
		if reset_seed == True:
			new_seed = np.random.randint(0, 32768)
			np.random.seed(new_seed)
			self.random_seed = new_seed

		if verbose:
			print "Generating tags. This may take awhile..."

		for i in np.arange(0, self.niter + 1): # generate some tags

			if verbose and i % 10000 == 0 and i > 0:
				print "Iteration: " + str(i) + "/" + str(self.niter)
				print "Tags found: ", len(self.master_list)/4

			if verbose and i == self.niter:
				print "Iteration: " + str(i) + "/" + str(self.niter)
				print "Tags found: ", len(self.master_list)/4

			new_tag = np.random.randint(0,2, size=(self.tag_len)).astype(np.uint8) # randomly generate a tag

			# get tag rotations
			tag_90 = utils.rotate_tag90(new_tag, self.tag_shape, 1) 
			tag_180 = utils.rotate_tag90(new_tag, self.tag_shape, 2)
			tag_270 = utils.rotate_tag90(new_tag, self.tag_shape, 3)
			tag_list = np.asarray([new_tag, tag_90, tag_180, tag_270]).astype(np.uint8)

			# check for differences between tag and rotations
			test = utils.check_diffs(tag_list, tag_list, self.ndiffs, 3)

			if test == 4: # if tag and all rotations are different enough from each other...

				if self.first_tag == True:

					self.master_list = tag_list
					self.first_tag = False # done with first tag

				elif self.first_tag == False:
					
					# check for differences between all tag rotations and master list of tags
					test = utils.check_diffs(tag_list, self.master_list, self.ndiffs, self.master_list.shape[0])
					
					if test == 4:  # if all tag rotations are different enough from master list...
						for tag in tag_list:
							tag = np.asarray([tag]).astype(np.uint8)
							self.master_list = np.append(self.master_list, tag, axis=0) 

		self.ntags = self.master_list.shape[0]/4

		ID_list = []

		for i in range(self.ntags):
			ID = [i,i,i,i]
			ID_list.append(ID)

		ID_array = np.array(ID_list) + 1
		self.id_list = ID_array.flatten()

		if verbose:
			print "Done!"
		
	def save_dict(self, filename = "master_list.pkl"):

		"""Save TagList configuration as ``.pkl`` file.

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
					"ndiffs" : self.ndiffs, 
					"white_width" : self.white_width, 
					"black_width" : self.black_width 
				 }

		output = open(filename, 'wb')

		try:
			pickle.dump(config, output)
		except:
			raise IOError("File must be '.pkl' extension")

		output.close()

		self.saved = True

	def load_dict(self, filename = "master_list.pkl"):

		"""Load TagList configuration from a ``.pkl`` file.

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
		self.ndiffs = config["ndiffs"]
		self.white_width = config["white_width"]
		self.black_width = config["black_width"]
		self.ntags = self.master_list.shape[0]/4
		self.tag_len = self.tag_shape[0]*self.tag_shape[1]

		self.loaded = True


	def print_tags(self, file, ntags = 200, page_size = (8.26, 11.69), ncols = 20, id_fontsize = 5, arrow_fontsize = 10, id_digits = 5, show = True):

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
		for i, tag in enumerate(self.master_list):
			if (i+1) % 4 == 0 and plot <= ntags:

				border_shape, tag = utils.add_border(tag, self.tag_shape, self.white_width, self.black_width)
				tag = tag.reshape(border_shape)

				ax = plt.subplot(int(ntags/ncols), ncols, plot)
				tag_number = str((i+1)/4)

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


