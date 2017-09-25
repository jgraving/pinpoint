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

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import pickle

from sklearn.metrics.pairwise import pairwise_distances


def rotate_tag90(tag, tag_shape, n_rot=1):

    """
    Rotate barcode tag 90 degrees.

    Parameters
    ----------
    tag : 1-D array_like
        Flattened barcode tag.
    tag_shape : tuple of int
        Shape of the barcode tag.
    n_rot : int
        Number of times to rotate 90 degrees.


    Returns
    -------
    rotated : 1-D array
        Returns rotated tag flattened to 1-D array.

    """

    tag = tag.reshape(tag_shape)
    rotated = np.rot90(tag, n_rot)
    rotated = rotated.flatten()

    return rotated


def add_border(tag, tag_shape, white_width=1, black_width=1):

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

    tag = tag.reshape(tag_shape)

    black_shape = (tag_shape[0] + (2 * white_width) + (2 * black_width),
                   tag_shape[1] + (2 * white_width) + (2 * black_width))
    black_border = np.zeros(black_shape, dtype=bool)

    white_shape = (tag_shape[0] + (2 * white_width),
                   tag_shape[1] + (2 * white_width))
    white_border = np.ones(white_shape, dtype=bool)

    white_idx = tag_shape[0] + white_width
    white_jdx = tag_shape[1] + white_width
    white_border[white_width:white_idx, white_width:white_jdx] = tag

    black_idx = tag_shape[0] + (2 * white_width) + black_width
    black_jdx = tag_shape[1] + (2 * white_width) + black_width
    black_border[black_width:black_idx, black_width:black_jdx] = white_border

    bordered_tag = black_border.flatten()

    return bordered_tag


def pairwise_distance_check(X, Y=None, distance=5, metric='cityblock'):
    """
    Check the pairwise distances between two arrays.

    Parameters
    ----------
    X : array
        Array of flattened barcodes.
    Y : array
        Array of flattened barcodes. default = None
    distance : float
        Minimum distance between all barcodes in X and Y.
    metric : str

    Returns
    -------
    test : bool
        Returns True if all elements are greater than the minimum distance.

    """
    D = pairwise_distances(X, Y, metric=metric)
    D[D < distance] = False
    D[D >= distance] = True

    if isinstance(Y, np.ndarray):
        if D.sum() == D.size:
            test = True
        else:
            test = False

    elif isinstance(Y, type(None)):
        if D.sum() == D.size - D.shape[0]:
            test = True
        else:
            test = False

    return test


def add_white_border(master_list, tag_shape, white_width):

    bordered = []

    bordered = [add_border(tag,
                           tag_shape,
                           white_width=white_width,
                           black_width=0)
                for tag in master_list]

    bordered = np.array(bordered)

    return bordered


def get_id_list(ntags):

    id_list = []
    for idx in range(ntags):
        ID = [idx, idx, idx, idx]
        id_list.append(ID)

    id_list = np.array(id_list) + 1
    id_list = id_list.flatten()

    return id_list


class TagDictionary:

    """
    A class for generating, saving, loading, and printing 2-D barcode tags.

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

    """

    def __init__(self, tag_shape=(5, 5), distance=7,
                 white_width=1, black_width=1, random_seed=None):

        self.tag_shape = tag_shape
        self.distance = distance
        self.white_width = white_width
        self.white_shape = tuple([x + (self.white_width * 2)
                                  for x in self.tag_shape])
        self.black_width = black_width
        self.black_shape = tuple([x + (self.white_width * 2) +
                                  (self.black_width * 2)
                                  for x in self.tag_shape])

        self.first_tag = True  # start with first tag

        # get length of flattened tag
        self.tag_len = self.tag_shape[0] * self.tag_shape[1]

        self.master_list = None
        self.loaded = False
        self.saved = False

        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def generate_dict(self, niter=99999, verbose=False):
        """
        Generate barcode tags.

        Parameters
        ----------
        niter : int, default = 99999
            Number of iterations to try.
        verbose : bool, default = False
            Prints updates when True.

        """

        if verbose:
            print("Generating tags. This may take awhile...")

        for idx in np.arange(0, niter + 1):  # generate some tags

            if verbose and (idx % 10000 == 0 or idx == niter) and idx > 0 \
               and not isinstance(self.master_list, type(None)):
                    print("Iteration: " + str(idx) + "/" + str(niter))
                    print("Tags found: ", len(self.master_list) // 4)

            # randomly generate a tag
            new_tag = np.random.randint(0, 2, size=(self.tag_len), dtype=bool)

            # get tag rotations
            tag_list = [rotate_tag90(new_tag, self.tag_shape, n_rot)
                        for n_rot in np.arange(1, 4)]
            tag_list = np.array(tag_list, dtype=bool)

            # check distance between tag and rotations
            test = pairwise_distance_check(tag_list, distance=self.distance)

            # if tag and all rotations are different enough from each other...
            if test is True:

                if self.first_tag is True:
                    self.master_list = tag_list
                    self.first_tag = False  # done with first tag

                elif self.first_tag is False:

                    # check distances with master list of tags
                    test = pairwise_distance_check(X=tag_list,
                                                   Y=self.master_list,
                                                   distance=self.distance)

                    # if tag different enough from master list...
                    if test is True:
                        for tag in tag_list:
                            tag = np.array([tag], dtype=bool)
                            self.master_list = np.append(self.master_list,
                                                         tag,
                                                         axis=0)

        self.ntags = self.master_list.shape[0] // 4

        self.id_list = get_id_list(self.ntags)

        if verbose:
            print("Done!")

        self.barcode_list = add_white_border(self.master_list,
                                             self.tag_shape,
                                             self.white_width)

        return self

    def save_dict(self, filename="master_list.pkl"):

        """
        Save configuration as ``.pkl`` file.

        Parameters
        ----------
        filename : str, default = "master_list.pkl"
            Path to save file, must be '.pkl' extension

        Returns
        -------
        saved : bool
            Saved successfully.
        """
        config = {"master_list": self.master_list,
                  "id_list": self.id_list,
                  "tag_shape": self.tag_shape,
                  "distance": self.distance,
                  "white_width": self.white_width,
                  "black_width": self.black_width,
                  }

        output = open(filename, 'wb')

        pickle.dump(config, output, protocol=0)

        output.close()

        self.saved = True

        return self.saved

    def load_dict(self, filename="master_list.pkl"):

        """
        Load configuration from a ``.pkl`` file.

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

        config = pickle.load(pkl_file)

        pkl_file.close()

        # Load new configuration
        self.master_list = config["master_list"]
        self.master_list = self.master_list.astype(bool)
        self.id_list = config["id_list"]
        self.tag_shape = config["tag_shape"]
        # self.distance = config["distance"]
        self.white_width = config["white_width"]
        self.black_width = config["black_width"]
        self.white_shape = tuple([x + (self.white_width * 2)
                                  for x in self.tag_shape])
        self.black_shape = tuple([x + (self.white_width * 2) +
                                  (self.black_width * 2)
                                  for x in self.tag_shape])
        self.ntags = self.master_list.shape[0] // 4
        self.id_index = np.arange(len(self.id_list))
        self.tag_len = self.tag_shape[0] * self.tag_shape[1]

        self.barcode_list = add_white_border(self.master_list,
                                             self.tag_shape,
                                             self.white_width)

        self.loaded = True

        return self.loaded

    def print_tags(self, filename, ntags=200,
                   page_size=(8.26, 11.69), ncols=20, id_fontsize=5,
                   arrow_fontsize=10, id_digits=4, show=True):

        """
        Print tags as image file or PDF.
        Default settngs are for ~6-mm wide tags.

        Parameters
        ----------
        filename : str
            Location for saving the barcode images,
            must be vector graphic (`.pdf`, '.svg', '.eps')
            or image (`.png`, `.jpg`, etc.) file extension.
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
            Number of digits for ID number
            printed below barcode (zero pads the left side of the number).
        show : bool
            Show the figure using plt.show()

        """

        self.fig = plt.figure(figsize=page_size)

        plot = 1

        for idx, tag in enumerate(self.master_list):
            if (idx + 1) % 4 == 0 and plot <= ntags:

                tag = add_border(tag,
                                 self.tag_shape,
                                 self.white_width,
                                 self.black_width)
                tag = tag.reshape(self.black_shape)

                ax = plt.subplot(ntags // ncols, ncols, plot)
                tag_number = str((idx + 1) / 4)

                if len(tag_number) < id_digits:
                    zero_pad = "0" * (id_digits - len(tag_number))
                    tag_number = zero_pad + tag_number

                ax.set_title(u'\u2191',
                             fontsize=arrow_fontsize,
                             family='sans-serif',
                             weight='normal'
                             )
                ax.set_xlabel(tag_number,
                              fontsize=id_fontsize,
                              family='sans-serif',
                              weight='heavy',
                              color='white',
                              backgroundcolor='black',
                              bbox={'fc': 'black', 'ec': 'none'}
                              )

                ax.set_aspect(1)
                ax.xaxis.set_ticks_position('none')
                ax.yaxis.set_ticks_position('none')
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.xaxis.set_label_coords(0.5, -0.1)

                ax.pcolormesh(tag,
                              cmap='gray',
                              zorder=200
                              )

                plot += 1

        plt.savefig(filename, interpolation='none')

        if show:
            plt.show()

        return True
