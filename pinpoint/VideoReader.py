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


class VideoReader:

    def __init__(self, source):

        self._source = source
        self.stream = cv2.VideoCapture(self._source)
        self.total_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.codec = self.stream.get(cv2.CAP_PROP_FOURCC)
        self.frame_height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.finished = False

    def read(self):

        ret, frame = self.stream.read()

        if ret:
            return frame
        else:
            self._finished = True

    def read_batch(self, batch_size=2, asarray=False):

        frames = []

        for idx in range(batch_size):
            ret, frame = self.stream.read()
            if ret:
                frames.append(frame)
            else:
                self.finished = True
                break

        if asarray and len(frames) > 0:
            frames = np.asarray(frames, dtype=np.uint8)

        return frames

    def close(self):
        self.stream.release()
        return not self.stream.isOpened()

    def current_frame(self, value=-1):
        if value < 0:
            return int(self.stream.get(cv2.CAP_PROP_POS_FRAMES))
        else:
            self.stream.set(cv2.CAP_PROP_POS_FRAMES, value)

    def current_time(self, value=-1):
        if value < 0:
            return self.stream.get(cv2.CAP_PROP_POS_MSEC)
        else:
            self.stream.set(cv2.CAP_PROP_POS_MSEC, value)

    @property
    def percent_finished(self):
        return (float(self.current_frame()) / float(self.total_frames)) * 100
