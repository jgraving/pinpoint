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
import multiprocessing


class Parallel:

    def __init__(self, n_jobs):

        if n_jobs < 0:
            n_jobs = multiprocessing.cpu_count() + n_jobs + 1
        elif n_jobs > multiprocessing.cpu_count():
            n_jobs = multiprocessing.cpu_count()

        self.n_jobs = n_jobs
        self.pool = multiprocessing.Pool(n_jobs)

    def process(self, job, arg, asarray=False):

        processed = self.pool.map(job, arg)

        if asarray:
            processed = np.asarray(processed)

        return processed

    def close(self):

        self.pool.close()
        self.pool.terminate()
        self.pool.join()
