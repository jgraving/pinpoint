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

import numpy as np
import imgstore


class StoreReader:

    def __init__(self, path):
        self.store = imgstore.new_for_filename(path)
        self.max = self.store.frame_max
        self.min = self.store.frame_min
        self.index = np.arange(self.min, self.max)
        self.store.frame_number = self.min

    def __len__(self):
        return self.max - self.min

    def get_data(self, indexes):
        indexes = self.index[indexes]

        frames = []
        frame_numbers = []
        frame_timestamps = []
        for idx in indexes:
            if idx is self.store.frame_number + 1:
                frame, (frame_number, frame_timestamp) = self.store.get_next_framenumber(exact_only=False)
            else:
                frame, (frame_number, frame_timestamp) = self.store.get_image(idx, exact_only=False)
            frames.append(frame)
            frame_numbers.append(frame_number)
            frame_timestamps.append(frame_timestamp)
        if len(indexes) > 1:
            frames = np.stack(frames)
            frame_numbers = np.array(frame_numbers)
            frame_timestamps = np.array(frame_timestamps)
        else:
            frames = frames[0]
            frame_numbers = frame_numbers[0]
            frame_timestamps = frame_timestamps[0]

        return frames, frame_numbers, frame_timestamps

    def _check_index(self, key):
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = len(self) - 1
            if stop < len(self):
                idx = range(start, stop)
            else:
                raise IndexError
        elif isinstance(key, (int, np.integer)):
            if key < len(self):
                idx = [key]
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) < len(self):
                idx = key.tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) < len(self):
                idx = key
            else:
                raise IndexError
        else:
            raise IndexError
        return idx

    def __getitem__(self, key):
        indexes = self._check_index(key)
        return self.get_data(indexes)
