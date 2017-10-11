![alt text][logo]

[logo]: logo-small.png

**pinpoint**: behavioral tracking using 2D barcode tags
=======================================

**pinpoint** is a Python library for generating and tracking 2D barcode tags. 
The library uses numpy and matplotlib to generate barcode tags and uses OpenCV to automatically track each tag. 
It provides a high-level API for the automated measurement of animal behavior and locomotion.

This software is still in early-release development. Expect some adventures. 

Citing
----------
If you use this software for academic research, please consider citing it using this zenodo DOI: 

[![DOI](https://zenodo.org/badge/89222910.svg)](https://zenodo.org/badge/latestdoi/89222910)


Installation
------------

Install the development version:
```bash
pip install git+https://www.github.com/jgraving/pinpoint.git
```
Install the latest stable release:
```bash
pip install https://github.com/jgraving/pinpoint/archive/v0.0.1-alpha.zip
```
Dependencies
------------

- [Python 2.7+](http://www.python.org)

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](http://matplotlib.org/)

- [scikit-learn](http://scikit-learn.org/stable/)

- [numba](http://numba.pydata.org/)

- [OpenCV 3.1+](http://opencv.org/)

Installing OpenCV
------------

OpenCV cannot be automatically installed using pip and must be installed separately. Here are instructions on how to accomplish this: 

[Instructions for MacOS](https://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/)

[Instructions for Ubuntu](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)

These instructions may be out-of-date.

Development
-------------
[https://github.com/jgraving/pinpoint](https://github.com/jgraving/pinpoint)

Please submit bugs or feature requests to the [GitHub issue tracker]((https://github.com/jgraving/pinpoint/issues/new)

License
------------

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/pinpoint/blob/master/LICENSE) for details.



