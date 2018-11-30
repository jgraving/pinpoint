    # -*- coding: utf-8 -*-
if __name__ == '__main__':
    import subprocess
    import sys

    subprocess.call([sys.executable, '-m', 'pip', 'install', '-U', 'git+https://github.com/jgraving/pinpoint.git', '--user'])

    from pinpoint import VideoReader, Tracker, StoreReader
    import matplotlib.pyplot as plt
    import time
    import h5py
    import glob
    import numpy as np
    import cv2


    store = '/mount/zfs-kn/recordings/kn-crec16/40_30c_20181129_121836/'
    barcodes = '/home/jgraving/pinpoint/barcodes/4x4_4bit/master_list.h5'
    block_size = 501
    offset = 50
    resize = 3
    clahe = (0.05, 300)

    reader = StoreReader(store, batch_size=200)

    tracker = Tracker(source=reader,
                      block_size=block_size,
                      offset=offset,
                      distance_threshold=8,
                      area_range=(30, 200),
                      tolerance=0.1,
                      channel=0,
                      otsu=True,
                      clahe=clahe,
                      resize=resize)


    tracker.load_dict(barcodes)
    tracker.track(store + 'barcode_tracker_output.h5', n_jobs=-1)

