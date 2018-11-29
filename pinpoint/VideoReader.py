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
import os


class VideoReader(cv2.VideoCapture):

    '''Read a video in batches.

    Parameters
    ----------
    path: str
        Path to the video file.
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

        if isinstance(path, str):
            if os.path.exists(path):
                super(VideoReader, self).__init__(path)
                self.path = path
            else:
                raise ValueError('file or path does not exist')
        else:
            raise TypeError('path must be str')
        self.batch_size = batch_size
        self.n_frames = int(self.get(cv2.CAP_PROP_FRAME_COUNT))
        if framerate:
            self.timestep = 1. / framerate
        else:
            self.timestep = 1.
        self.idx = 0
        self.fps = self.get(cv2.CAP_PROP_FPS)
        self.height = self.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width = self.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.shape = (self.height, self.width)
        self.finished = False
        self.gray = gray
        self._read = super(VideoReader, self).read

    def read(self):
        ''' Read one frame

        Returns
        -------
        frame: array
            Image is returned of the frame if a frame exists.
            Otherwise, return None.

        '''
        ret, frame = self._read()
        if ret:
            if self.gray:
                frame = frame[..., 1][..., None]
            self.idx += 1
            return self.idx - 1, frame
        else:
            self.finished = True
            return None

    def read_batch(self):
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
        for idx in range(self.batch_size):
            frame = self.read()
            if frame is not None and not self.finished:
                frame_idx, frame = frame
                frames.append(frame)
                frames_idx.append(frame_idx)
        empty = len(frames) == 0
        if not empty:
            frames = np.stack(frames)
            frames_idx = np.array(frames_idx)
            timestamps = frames_idx * self.timestep
            return frames, frames_idx, timestamps
        else:
            return None

    def close(self):
        ''' Close the VideoReader.

        Returns
        -------
        bool
            Returns True if successfully closed.

        '''
        self.release()
        return not self.isOpened()

    def __len__(self):
        return int(np.ceil(self.n_frames / float(self.batch_size)))

    def __getitem__(self, index):

        if self.finished:
            raise StopIteration
        if isinstance(index, (int, np.integer)):
            idx0 = index * self.batch_size
            if self.idx != idx0:
                self.set(cv2.CAP_PROP_POS_FRAMES, idx0 - 1)
                self.idx = idx0
        else:
            raise NotImplementedError

        return self.read_batch()

    def __next__(self):

        if self.finished:
            raise StopIteration
        else:
            return self.read_batch()

    def __del__(self):
        self.close()

    @property
    def current_frame(self):
        return int(self.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def current_time(self):
        return self.get(cv2.CAP_PROP_POS_MSEC)

    @property
    def percent_finished(self):
        return self.get(cv2.CAP_PROP_POS_AVI_RATIO) * 100
