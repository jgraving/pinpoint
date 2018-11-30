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


class StoreFrameReader:

    def __init__(self, path):
        self.store = imgstore.new_for_filename(path)
        self.metadata = self.store.get_frame_metadata()
        self.metaindex = np.array(self.metadata['frame_number'])
        self.shape = self.store.image_shape

    def __len__(self):
        return self.metaindex.shape[0]

    def get_data(self, indexes):
        indexes = self.metaindex[indexes]

        frames = []
        frames_idx = []
        timestamps = []
        for idx in indexes:
            if idx is self.store.frame_number + 1:
                frame, (frame_idx, timestamp) = self.store.get_next_framenumber(exact_only=False)
            else:
                frame, (frame_idx, timestamp) = self.store.get_image(idx, exact_only=False)
            frames.append(frame)
            frames_idx.append(frame_idx)
            timestamps.append(timestamp)
        frames = np.stack(frames)
        if frames.ndim == 3:
            frames = frames[..., np.newaxis]
        frames_idx = np.array(frames_idx)
        timestamps = np.array(timestamps)

        return frames, frames_idx, timestamps

    def _check_index(self, key):
        if isinstance(key, slice):
            start = key.start
            stop = key.stop
            if start is None:
                start = 0
            if stop is None or stop >= len(self):
                stop = len(self)
            if stop <= len(self):
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


class StoreReader:

    def __init__(self, path, batch_size):
        self.store = StoreFrameReader(path)
        self.shape = self.store.shape
        self.batch_size = batch_size
        self.n_frames = len(self.store)
        self.idx = 0

    def __len__(self):
        return int(np.ceil(self.n_frames / float(self.batch_size)))

    def get_batch(self, index=None):
        if index:
            if isinstance(index, (int, np.integer)):
                if index < 0:
                    index += len(self)
                if index < len(self):
                    idx0 = index * self.batch_size
                    idx1 = idx0 + self.batch_size
                    self.idx = index
                else:
                    raise IndexError('index out of range')
            else:
                raise NotImplementedError
        else:
            if self.idx >= len(self) - 1:
                self.idx = 0
            if self.idx < len(self) - 1:
                idx0 = self.idx * self.batch_size
                idx1 = idx0 + self.batch_size
                self.idx += 1
            else:
                raise StopIteration

        return self.store[idx0:idx1]

    def __getitem__(self, index):
        return self.get_batch(index)