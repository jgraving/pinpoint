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

from .TagDictionary import TagDictionary
from .CameraCalibration import CameraCalibration
from .VideoReader import VideoReader
from .Parallel import Parallel

from sklearn.neighbors import NearestNeighbors

import h5py

from .utils import process_frame, tag_template

import gc

# disable multithreading in OpenCV for main thread
# to avoid problems after parallelization
cv2.setNumThreads(0)

__all__ = ['Tracker']


def _process_frames_parallel(feed_dict):

    frame = feed_dict["frame"]
    channel = feed_dict["channel"]
    resize = feed_dict["resize"]
    block_size = feed_dict["block_size"]
    offset = feed_dict["offset"]
    barcode_shape = feed_dict["barcode_shape"]
    white_width = feed_dict["white_width"]
    area_min = feed_dict["area_min"]
    area_max = feed_dict["area_max"]
    x_proximity = feed_dict["x_proximity"]
    y_proximity = feed_dict["y_proximity"]
    tolerance = feed_dict["tolerance"]
    max_side = feed_dict["max_side"]
    template = feed_dict["template"]
    var_thresh = feed_dict["var_thresh"]
    barcode_nn = feed_dict["barcode_nn"]
    id_list = feed_dict["id_list"]
    id_index = feed_dict["id_index"]
    distance_threshold = feed_dict["distance_threshold"]

    fetch_dict = process_frame(frame=frame,
                               channel=channel,
                               resize=resize,
                               block_size=block_size,
                               offset=offset,
                               barcode_shape=barcode_shape,
                               white_width=white_width,
                               area_min=area_min,
                               area_max=area_max,
                               x_proximity=x_proximity,
                               y_proximity=y_proximity,
                               tolerance=tolerance,
                               max_side=max_side,
                               template=template,
                               var_thresh=var_thresh,
                               barcode_nn=barcode_nn,
                               id_list=id_list,
                               id_index=id_index,
                               distance_threshold=distance_threshold
                               )

    fetch_dict = {"points_array": fetch_dict["points_array"],
                  "best_id_list": fetch_dict["best_id_list"],
                  "distances": fetch_dict["distances"]
                  }

    return fetch_dict


def process_frames_parallel(frames, channel, resize,
                            block_size, offset, barcode_shape,
                            white_width, area_min, area_max,
                            x_proximity, y_proximity, tolerance,
                            template, max_side, var_thresh,
                            barcode_nn, id_list, id_index,
                            distance_threshold, pool):

    feed_dicts = [{"frame": frame,
                   "channel": channel,
                   "resize": resize,
                   "block_size": block_size,
                   "offset": offset,
                   "barcode_shape": barcode_shape,
                   "white_width": white_width,
                   "area_min": area_min,
                   "area_max": area_max,
                   "x_proximity": x_proximity,
                   "y_proximity": y_proximity,
                   "tolerance": tolerance,
                   "max_side": max_side,
                   "template": template,
                   "var_thresh": var_thresh,
                   "barcode_nn": barcode_nn,
                   "id_list": id_list,
                   "id_index": id_index,
                   "distance_threshold": distance_threshold}
                  for frame in frames]

    del frames
    gc.collect()

    fetch_dicts = pool.process(_process_frames_parallel,
                               feed_dicts, asarray=False)

    return fetch_dicts


class Tracker(TagDictionary, VideoReader, CameraCalibration):
    """

    Tracker class for processing videos to track barcodes.

    Parameters
    ----------
    block_size : int, default = 1001
        Odd value integer.
        Size of the local neighborhood
        for adaptive thresholding.

    offset : default = 2
        Constant subtracted from
        the mean for adaptive thresholding.
        Normally, it is positive but
        may be zero or negative as well.
        The threshold value is calculated
        as the mean of the block_size x block_size
        neighborhood *minus* the offset.

    area_range : tuple, default (10,10000)
        Area range in pixels for potential barcodes.
        If the minimum value is too low this
        can lead to false positives.

    tolerance : int or float, default = 0.1
        This parameter affects how many many contours
        reach the barcode matching algorithm,
        as only polygons with 4 vertices are used.
        This is the tolerance for fitting a polygon as a
        proportion of the perimeter of the contour.
        This value is used to set epsilon, which is the
        maximum distance between the original contour
        and its polygon approximation. Higher values
        decrease the number of vertices in the polygon.
        Lower values increase the number of vertices in
        the polygon.

    distance_threshold : int, default = 8
        The maximum Hamming distance between a
        barcode candidate and its matched identity.
        Set this to some high value
        to save all candidates.

    var_thresh : float, (default = 500)
        Minimum variance threshold.
        Candidate barcodes with low variance
        are likely white or black blobs.

    channel : {'blue', 'green', 'red', 'none', None}, default = None
        The color channel to use for
        producing the grayscale image.

    resize : float, default=1.0
        A scalar value for resizing images.
        In most cases, increasing the size of the image
        can improve edge detection which leads
        to better barcode reconstruction
        at the expense of computation time.
        The recommended setting is
        some value between 1.0 and 2.0

    Returns
    -------
    self : class
        Tracker class instance

    """

    def __init__(self, source, block_size=1001, offset=80,
                 area_range=(10, 10000), tolerance=0.1, distance_threshold=8,
                 var_thresh=500, channel='green', resize=1.0):

        VideoReader.__init__(self, source)
        TagDictionary.__init__(self)
        CameraCalibration.__init__(self)
        self.area_min = area_range[0] * (resize**2)
        self.area_max = area_range[1] * (resize**2)
        self.tolerance = tolerance
        self.distance_threshold = distance_threshold
        self.block_size = block_size
        self.offset = offset
        self.channel = channel
        self.var_thresh = var_thresh
        self.resize = resize
        self.x_proximity = (self.frame_width * self.resize) - 1
        self.y_proximity = (self.frame_height * self.resize) - 1

    def track(self, filename='output.h5', batch_size=8, n_jobs=1):

        """
        Process frames to track barcodes.
        Saves data to HDF5 file. See `Notes` for details.

        Parameters
        ----------
        filename : str, default = 'output.h5'
            The output file for saving data.
        batch_size : int, default is 8
            The number of frames to process in each batch.
        n_jobs : int (default = 1)
            Number of jobs to use for processing images on the CPU.
            If -1, all CPUs are used. If 1 is given, no parallel computing is
            used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            Thus for n_jobs = -2, all CPUs but one are used.

        Notes
        -----
        The tracker outputs data as an HDF5 file with the following structure:
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
        fetch_dict : dict

        A dictionary containing the following objects
        processed from the latest frame:

        "gray" : ndarray, shape (MxNx1)
            The grayscale image. Only returned if n_jobs=1
        "thresh" : ndarray, shape (MxNx1)
            The threshold image. Only returned if n_jobs=1
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
        if n_jobs == 0:
            n_jobs = 1

        self.n_jobs = n_jobs

        self.h5file = h5py.File(filename, 'w')

        self.h5file.attrs.create('fps', self.fps)
        self.h5file.attrs.create('codec', self.codec)
        self.h5file.attrs.create('height', self.frame_height)
        self.h5file.attrs.create('width', self.frame_width)
        self.h5file.attrs.create('total_frames', self.total_frames)
        # self.h5file.attrs.create('source', self._source)

        data_group = self.h5file.create_group('data')
        frame_idx_dset = data_group.create_dataset('frame_idx',
                                                   shape=(0,),
                                                   dtype=np.int64,
                                                   maxshape=(None,))
        corners_dset = data_group.create_dataset('corners',
                                                 shape=(0, 4, 2),
                                                 dtype=np.float64,
                                                 maxshape=(None, 4, 2))
        identity_dset = data_group.create_dataset('identity',
                                                  shape=(0,),
                                                  dtype=np.int32,
                                                  maxshape=(None,))
        distances_dset = data_group.create_dataset('distances',
                                                   shape=(0,),
                                                   dtype=np.int32,
                                                   maxshape=(None,))

        dset_list = [frame_idx_dset,
                     corners_dset,
                     identity_dset,
                     distances_dset]

        self.barcode_shape = self.white_shape
        max_side = 100
        template = tag_template(max_side)

        self.barcode_nn = NearestNeighbors(metric='cityblock',
                                           algorithm='ball_tree')
        self.barcode_nn.fit(self.barcode_list)

        idx = 0

        if self.n_jobs != 1:
            self.pool = Parallel(self.n_jobs)

        try:
            while not self.finished:

                frames = self.read_batch(batch_size)

                if self.n_jobs == 1:
                    fetch_dicts = [process_frame(
                                   frame=frame,
                                   channel=self.channel,
                                   resize=self.resize,
                                   block_size=self.block_size,
                                   offset=self.offset,
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
                                   distance_threshold=self.distance_threshold
                                   ) for frame in frames]

                else:

                    fetch_dicts = process_frames_parallel(
                        frames=frames,
                        channel=self.channel,
                        resize=self.resize,
                        block_size=self.block_size,
                        offset=self.offset,
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
                        pool=self.pool
                    )

                for fetch_dict in fetch_dicts:

                    points_array = fetch_dict["points_array"]
                    best_id_list = fetch_dict["best_id_list"]
                    distances = fetch_dict["distances"]

                    points_array = points_array / self.resize
                    frame_idx = np.repeat(idx, points_array.shape[0])
                    idx += 1
                    data_list = [frame_idx,
                                 points_array,
                                 best_id_list,
                                 distances]

                    for (dset, data) in zip(dset_list, data_list):

                        current_shape = list(dset.shape)
                        current_size = current_shape[0]

                        new_shape = current_shape
                        new_shape[0] = new_shape[0] + data.shape[0]
                        new_size = new_shape[0]

                        dset.resize(tuple(new_shape))
                        dset[current_size:new_size] = data

            if self.n_jobs != 1:
                self.pool.close()

            self.h5file.close()

        except KeyboardInterrupt:
            if self.n_jobs != 1:
                self.pool.close()
            self.h5file.close()

        return True
