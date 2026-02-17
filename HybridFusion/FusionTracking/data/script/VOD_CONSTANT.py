"""
Public constant accessible to all files
"""

import numpy as np

# metrics with two return values
METRIC = ['giou_bev', 'feat_similarity']
FAST_METRIC = ['giou_bev']
SECOND_METRIC = ['feat_similarity']

# category name(str) <-> category label(int)
CLASS_SEG_TO_STR_CLASS = {'car': 0, 'pedestrian': 1, 'cyclist': 2}
CLASS_STR_TO_SEG_CLASS = {0: 'car', 1: 'pedestrian', 2: 'cyclist'}

# math
PI, TWO_PI = np.pi, 2 * np.pi

# init UKFP for different non-linear motion model
CTRA_INIT_UFKP = {
    # [x, y, z, w, l, h, v, a, theta, omega]
    'car': [4, 4, 4, 4, 4, 4, 100, 4, 1, 0.1],
}
CTRV_INIT_UFKP = {
    # [x, y, z, w, l, h, v, theta, omega]
    'pedestrian': [10, 10, 10, 10, 10, 10, 10, 100, 10]
}
BIC_INIT_UKFP = {
    # [x, y, z, w, l, h, v, a, theta, sigma]
    'cyclist': [10, 10, 10, 10, 10, 10, 1000, 10, 10, 10]
}