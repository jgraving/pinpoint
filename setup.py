#! /usr/bin/env python
#
# Copyright (C) 2015-2016 Jacob Graving <jgraving@gmail.com>

import os
# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."

DESCRIPTION = "pinpoint: behavioral tracking using 2D barcode tags"
LONG_DESCRIPTION = """\
pinpoint is a Python library for generating and tracking 2D barcode tags. 
The library uses numpy and matplotlib to generate barcode tags and uses OpenCV to automatically track each tag. 
It provides a high-level API for the automated measurement of animal behavior and locomotion.
"""

DISTNAME = 'pinpoint'
MAINTAINER = 'Jacob Graving'
MAINTAINER_EMAIL = 'jgraving@gmail.com'
URL = 'http://jakegraving.com'
LICENSE = 'Apache Software License 2.0'
DOWNLOAD_URL = 'https://jgraving@github.com/jgraving/pinpoint.git'
VERSION = '0.0.3-dev'

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

def check_dependencies():
    install_requires = []

    # Make sure dependencies exist

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')
    try:
        import matplotlib
    except ImportError:
        install_requires.append('matplotlib')
    try:
        import cv2
    except ImportError:
        install_requires.append('opencv-python')
    try:
        import pickle
    except ImportError:
        install_requires.append('pickle')
    try:
        import glob
    except ImportError:
        install_requires.append('glob')
    try:
        import sklearn
    except ImportError:
        install_requires.append('sklearn')
    try:
        import h5py
    except ImportError:
        install_requires.append('h5py')
    try:
        import numba
    except ImportError:
        install_requires.append('numba')
    try:
        import deepdish
    except ImportError:
        install_requires.append('deepdish')
    try:
        import types
    except ImportError:
        install_requires.append('types')
    try:
        import warnings
    except ImportError:
        install_requires.append('warnings')
    try:
        import tqdm
    except ImportError:
        install_requires.append('tqdm')
    try:
        import imgstore
    except ImportError:
        install_requires.append('imgstore')

    return install_requires

if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        install_requires=install_requires,
        packages=['pinpoint'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3',
                     'License :: OSI Approved :: Apache Software License',
                     'Topic :: Scientific/Engineering :: Visualization',
                     'Topic :: Scientific/Engineering :: Image Recognition',
                     'Topic :: Scientific/Engineering :: Information Analysis',
                     'Topic :: Multimedia :: Video'
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
          )
