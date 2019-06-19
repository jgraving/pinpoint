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

Dependencies
------------

- [Python 3.5+](http://www.python.org) 

- [numpy](http://www.numpy.org/)

- [scipy](http://www.scipy.org/)

- [matplotlib](http://matplotlib.org/)

- [scikit-learn](http://scikit-learn.org/stable/)

- [numba](http://numba.pydata.org/)

- [OpenCV 3.1+](http://opencv.org/)

Development
-------------
[https://github.com/jgraving/pinpoint](https://github.com/jgraving/pinpoint)

Please submit bugs or feature requests to the [GitHub issue tracker](https://github.com/jgraving/pinpoint/issues/new)

License
------------

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/pinpoint/blob/master/LICENSE) for details.

References 
------------
pinpoint is based on [ArUco](https://www.uco.es/investiga/grupos/ava/node/26):

```
Garrido-Jurado, S., Muñoz-Salinas, R., Madrid-Cuevas, F. J., & Marín-Jiménez, M. J. (2014). Automatic generation and detection of highly reliable fiducial markers under occlusion. Pattern Recognition, 47(6), 2280-2292.
```

Other similar marker systems are also publicly available, such as [AprilTag](https://april.eecs.umich.edu/software/apriltag.html):

```
Wang, J., & Olson, E. (2016). AprilTag 2: Efficient and robust fiducial detection. In Intelligent Robots and Systems (IROS), 2016 IEEE/RSJ International Conference (pp. 4193-4198). IEEE.
```

If you require barcode tracking in MATLAB see [BEETag](https://github.com/jamescrall/BEEtag):

```
Crall, J. D., Gravish, N., Mountcastle, A. M., & Combes, S. A. (2015). BEEtag: a low-cost, image-based tracking system for the study of animal behavior and locomotion. PloS one, 10(9), e0136487.
```

