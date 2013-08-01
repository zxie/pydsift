#cython: boundscheck=False, wraparound=False, profile=False

from __future__ import division

import numpy as np
cimport numpy as np

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

cdef extern from "math.h":
    double sqrt(double x)

def normalize_sift(np.ndarray[np.float_t, ndim=2] descs):
    '''
    Does sift normalization (normalize to 1, threshold at 0.2,
    renormalize)
    Also does high-contrast thresholding
    '''
    SIFT_THRES = 0.2
    NORM_THRES = 1.0  # minimum normalization denominator

    cdef int i, desc_idx, feat_idx
    cdef np.float_t norm, value, new_norm

    cdef int num_descs = descs.shape[0]
    cdef int num_feats = descs.shape[1]

    for desc_idx in range(num_descs):
        norm = 0
        for feat_idx in range(num_feats):
            norm += descs[desc_idx, feat_idx] ** 2
        norm = sqrt(norm)
        if norm > NORM_THRES:
            new_norm = 0
            for feat_idx in range(num_feats):
                value = descs[desc_idx, feat_idx] / norm
                if value > SIFT_THRES:
                    value = SIFT_THRES
                descs[desc_idx, feat_idx] = value
                new_norm += value * value
            new_norm = sqrt(new_norm)
            for feat_idx in range(num_feats):
                descs[desc_idx, feat_idx] /= new_norm

    return descs

def conv2(np.ndarray[np.float_t, ndim=2] f, np.ndarray[np.float_t, ndim=2] g):

    if g.shape[0] % 2 != 1 or g.shape[1] % 2 != 1:
        raise ValueError("Only odd dimensions on filter supported")

    assert f.dtype == np.float and g.dtype == np.float

    cdef int vmax = f.shape[0]
    cdef int wmax = f.shape[1]
    cdef int smax = g.shape[0]
    cdef int tmax = g.shape[1]
    cdef int smid = smax // 2
    cdef int tmid = tmax // 2
    cdef int xmax = vmax + 2*smid
    cdef int ymax = wmax + 2*tmid
    cdef np.ndarray[np.float_t, ndim=2] h = np.empty([xmax, ymax], dtype=np.float)
    cdef int x, y, s, t, v, w

    cdef int s_from, s_to, t_from, t_to

    cdef double value

    for x in range(xmax):
        for y in range(ymax):
            s_from = int_max(smid - x, -smid)
            s_to = int_min((xmax - x) - smid, smid + 1)
            t_from = int_max(tmid - y, -tmid)
            t_to = int_min((ymax - y) - tmid, tmid + 1)
            value = 0
            for s in range(s_from, s_to):
                for t in range(t_from, t_to):
                    v = x - smid + s
                    w = y - tmid + t
                    value += g[smid - s, tmid - t] * f[v, w]
            h[x, y] = value
    return h[smid:-smid, tmid:-tmid]