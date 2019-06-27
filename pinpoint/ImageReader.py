#! /usr/bin/env python

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

import cv2
import numpy as np
import glob


class ImageReader:

    '''Read images in batches.

    Parameters
    ----------
    path: str
        Glob path to the images.
    batch_size: int, default = 1
        Batch size for reading frames
    framerate: float, default = None
        Video framerate for determining timestamps
        for each frame. If None, timestamps will
        equal frame number.
    gray: bool, default = False
        If gray, return only the middle channel
    '''

    def __init__(self, path, batch_size=1, framerate=None, gray=False):

        #if isinstance(path, str):
        #    if os.path.exists(path):
        #        self.path = path
        #    else:
        #        raise ValueError('file or path does not exist')
        #else:
        #    raise TypeError('path must be str')
        self.path = path
        self.image_paths = glob.glob(path)
        self.batch_size = batch_size
        self.n_frames = len(self.image_paths)
        if framerate:
            self.timestep = 1. / framerate
        else:
            self.timestep = 1.
        test_images = cv2.imread(self.image_paths[0])
        self.height = test_images.shape[0]
        self.width = test_images.shape[1]
        self.shape = (self.height, self.width)
        self.gray = gray
        self.idx = 0

    def read(self, idx):
        ''' Read one frame

        Returns
        -------
        frame: array
            Image is returned of the frame if a frame exists.
            Otherwise, return None.

        '''
        frame = cv2.imread(self.image_paths[idx])
        if self.gray:
            frame = frame[..., 1][..., None]
        return idx, frame

    def read_batch(self, idx0, idx1):
        ''' Read in a batch of frames.

        Returns
        -------
        frames_idx: array
            A batch of frames from the video.

        frames: array
            A batch of frames from the video.

        '''
        frames = []
        frames_idx = []
        for idx in range(idx0, idx1):
            frame = self.read(idx)
            frame_idx, frame = frame
            frames.append(frame)
            frames_idx.append(frame_idx)
        if len(frames) == 1:
            frames = frames[0][None,]
            frames_idx = np.array(frames_idx)
            timestamps = frames_idx * self.timestep
        elif len(frames) > 1:
            frames = np.stack(frames)
            frames_idx = np.array(frames_idx)
            timestamps = frames_idx * self.timestep
        return frames, frames_idx, timestamps

    def __len__(self):
        return int(np.ceil(self.n_frames / float(self.batch_size)))

    def __getitem__(self, index):

        if isinstance(index, (int, np.integer)):
            idx0 = index * self.batch_size
            idx1 = (index + 1) * self.batch_size
        else:
            raise NotImplementedError

        return self.read_batch(idx0, idx1)

    def __next__(self):
        if self.idx < len(self):
            output = self.__getitem__(self.idx)
            self.idx += 1
            return output
        else:
            self.idx = 0
            StopIteration
