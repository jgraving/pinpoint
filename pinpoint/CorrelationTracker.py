import dlib


class CorrelationTracker():

    """

    CorrelationTracker class for tracking between barcodes
    using correlation filters [1] implemented in dlib.

    Parameters
    ----------
    frame : array_like, shape = (h, w, 3)
        the starting frame for tracking
    centroid : iterable, shape = (2,)
        x,y-coordinates for the starting point
    bbox_size : int, default = 50
        bounding box size in pixels for the correlation tracker

    References
    ----------
    [1] Danelljan, Martin, et al. "Accurate scale estimation for
        robust visual tracking." Proceedings of the British Machine
        Vision Conference BMVC. 2014.

    """

    def __init__(self, frame, centroid, bbox_size):

        row = centroid[0] - bbox_size // 2
        column = centroid[1] - bbox_size // 2
        bbox = (column, row, column + bbox_size, row + bbox_size)
        self.tracker = dlib.correlation_tracker()
        self.tracker.start_track(frame, dlib.rectangle(*bbox))

    def track(self, frame):

        score = self.tracker.update(frame)
        bbox = self.tracker.get_position()

        left = bbox.left()
        top = bbox.top()
        right = bbox.right()
        bottom = bbox.bottom()

        column = (left + right) / 2
        row = (top + bottom) / 2

        centroid = [row, column]

        return (score, centroid)
