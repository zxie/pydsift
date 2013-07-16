'''
Dense SIFT implementation based on Svetlana Lazebnik and Yangqing Jia's code
- http://www.cs.illinois.edu/homes/slazebni/research/SpatialPyramid.zip
- https://github.com/Yangqing/dsift-python
'''

import numpy as np
from scipy.signal import convolve2d
from scipy.misc import lena
#import matplotlib.pyplot as plt

# Flag to compare against Lazebnik's MATLAB implementation
# Should match exactly for SpatialPyramid.zip updated 2/29/2012
DEBUG = False

if DEBUG:
    from mlabwrap import mlab


def gauss_filt(sigma):
    # Isotropic
    w = 2 * int(np.ceil(sigma))
    G = np.array(xrange(-w, w + 1)) ** 2
    G = G.reshape((G.size, 1)) + G
    G = np.exp(-G / (2.0 * sigma * sigma))
    G /= np.sum(G)
    return G


def dgauss_filt(sigma):
    '''
    Generate a derivative of Gaussian filter in x (left-to-right)
    and y (top-to-bottom) directions
    '''
    G = gauss_filt(sigma)
    G_y, G_x = np.gradient(G)

    G_x *= 2.0 / np.sum(np.abs(G_x))
    G_y *= 2.0 / np.sum(np.abs(G_y))

    return G_x, G_y


class DenseSIFTExtractor:

    def __init__(self,
                 grid_spacing=1,
                 patch_size=16,
                 num_angles=8,
                 num_bins=4,
                 alpha=9.0,
                 sigma=1.0,
                 ori=0,
                 signed=True):
        '''
        grid_spacing: specifies how densely to compute descriptors
          patch_size: size of patches over which to compute descriptors
          num_angles: number of angles in sift histograms
            num_bins: number of bins along x and y of patches,
                      num_bins^2 * num_angles = sift descriptor size (128)
               alpha: parameter for attenuation of angles (must be odd)
               sigma: standard deviation for Gaussian smoothing before
                      computing gradients
                 ori: start orientation (in degrees), counter-clockwise
        '''
        self.grid_spacing = grid_spacing
        self.patch_size = patch_size
        self.num_angles = num_angles
        self.num_bins = num_bins
        self.alpha = alpha
        self.sigma = sigma
        self.signed = signed
        assert ori in (0, 90, 180, 270),\
                'Only ori in {0, 90, 180, 270} currently supported'
        self.ori = ori
        if DEBUG:
            assert ori == 0, 'DEBUG comparison only works for ori=0'

    def __str__(self):
        return 'DSIFT'

    def is_dense(self):
        return True

    def get_padding(self):
        return int(self.patch_size / 2) - 1

    #@profile
    def extract_descriptors(self, img, normalize=True, flatten=True):
        img = img.astype(np.double)
        imshape = img.shape

        if self.ori != 0:
            img = np.rot90(img, k=self.ori / 90)

        if DEBUG:
            descs0, _, _ = mlab.sp_dense_sift(img, self.grid_spacing,
                    self.patch_size, nout=3)
            descs0 = mlab.double(descs0)

        # Convert to grayscale
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        rows, cols = img.shape

        # Normalize to 1
        img /= np.max(np.max(img))

        # Add padding
        img = np.vstack((np.flipud(img[0:2, :]), img, np.flipud(img[-2:, :])))
        img = np.hstack((np.fliplr(img[:, 0:2]), img,\
                np.fliplr(img[:, -2:])))

        # Subtract away mean
        img -= np.mean(img)

        # Smooth
        G_x, G_y = dgauss_filt(self.sigma)
        I_x = -convolve2d(img, G_x, mode='same')
        I_y = -convolve2d(img, G_y, mode='same')
        I_x = I_x[2:-2, 2:-2]
        I_y = I_y[2:-2, 2:-2]

        # Calculate gradient magnitude and orientations
        I_mag = np.sqrt(I_x ** 2 + I_y ** 2)
        I_theta = np.arctan2(I_y, I_x)
        I_theta[np.isnan(I_theta)] = 0
        if not self.signed:
            I_theta = (I_theta + np.pi) % np.pi
            max_angle = np.pi
        else:
            max_angle = 2 * np.pi

        angles = np.arange(0, max_angle, max_angle / self.num_angles)

        # Orientation image (direction and magnitude)
        I_ori = np.zeros([rows, cols, self.num_angles])
        I_cos = np.cos(I_theta)
        I_sin = np.sin(I_theta)

        for k in xrange(self.num_angles):
            tmp = np.maximum((I_cos * np.cos(angles[k]) + I_sin *\
                    np.sin(angles[k])) ** self.alpha, 0)
            I_ori[:, :, k] = tmp * I_mag

        margin = self.patch_size / 2
        cx = margin - 0.5
        sample_res = self.patch_size / float(self.num_bins)
        weight_x = abs(np.arange(self.patch_size) + 1 - cx) / sample_res
        weight_x = (1 - weight_x) * (weight_x <= 1)

        wr = weight_x.reshape(1, -1)
        wc = weight_x.reshape(-1, 1)
        for k in xrange(self.num_angles):
            tmp = convolve2d(I_ori[:, :, k], wr)
            tmp = convolve2d(tmp, wc)
            offset = (tmp.shape[0] - I_ori.shape[0]) / 2.0
            I_ori[:, :, k] = tmp[np.ceil(offset):np.ceil(-offset),
                    np.ceil(offset):np.ceil(-offset)]

        tmp = np.linspace(0, self.patch_size, self.num_bins + 1).astype(int)
        sample_y, sample_x = np.meshgrid(tmp, tmp)  # Order on purpose
        sample_x = sample_x[0:self.num_bins, 0:self.num_bins].flatten()
        sample_y = sample_y[0:self.num_bins, 0:self.num_bins].flatten()
        sample_x -= self.patch_size / 2
        sample_y -= self.patch_size / 2

        # Construct the feature array and indices
        grid_x = np.arange(margin, cols - margin + 2,
                self.grid_spacing, dtype=int)
        grid_y = np.arange(margin, rows - margin + 2,
                self.grid_spacing, dtype=int)

        descs = np.zeros((grid_y.size, grid_x.size, self.num_angles *\
                self.num_bins ** 2))
        b = 0
        for k in xrange(self.num_bins ** 2):
            inds_x = grid_x + sample_x[k]
            inds_y = grid_y + sample_y[k]
            descs[:, :, b:b + self.num_angles] =\
                    I_ori.take(inds_y, axis=0).take(inds_x, axis=1)
            b += self.num_angles

        if self.ori != 0:
            descs = np.rot90(descs, k=4 - self.ori / 90)

        [nrows, ncols, cols] = descs.shape
        descs = np.reshape(descs, [nrows * ncols, self.num_angles *\
                self.num_bins ** 2], order='F')
        if normalize:
            descs = self.normalize_sift(descs.T)
        else:
            descs = descs.T

        if DEBUG:
            descs = descs.T
            descs = np.reshape(descs, [nrows, ncols, self.num_angles *\
                    self.num_bins ** 2], order='F')
            print 'Difference:'
            diff = descs - descs0
            print diff
            print descs.shape
            print descs0.shape
            print 'max diff: ', np.max(np.abs(diff))
            return

        # TODO Way to remove this part?
        descs = descs.T
        descs = np.reshape(descs, [nrows, ncols, self.num_angles *\
                self.num_bins ** 2], order='F')

        if flatten:
            s = descs.shape
            descs = np.reshape(descs, (s[0] * s[1], s[2]))
            return descs.T, self.get_indices(imshape)
        else:
            return descs

    def get_indices(self, imshape):
        margin = self.get_padding()
        rowcol2ind = np.zeros(([imshape[0], imshape[1]]))
        idx = 0
        for i in xrange(imshape[0]):
            for j in xrange(imshape[1]):
                if i < margin or j < margin or \
                    i > imshape[0] - margin - 1 or j > imshape[1] - margin - 1:
                    rowcol2ind[i, j] = -1
                else:
                    rowcol2ind[i, j] = (i - margin) *\
                            (imshape[1] - margin * 2) + (j - margin)
                    idx += 1
        return rowcol2ind

    #@profile
    def normalize_sift(self, descs):
        '''
        Does sift normalization (normalize to 1, threshold at 0.2,
        renormalize)
        Also does high-contrast thresholding
        '''
        descs = np.array(descs, dtype=np.float64)

        SIFT_THRES = 0.2
        NORM_THRES = 1.0  # minimum normalization denominator

        norms = np.sqrt(np.sum(descs ** 2, axis=0))
        hcontrast_inds = norms > NORM_THRES
        descs_norm = descs[:, hcontrast_inds]
        descs_norm /= norms[hcontrast_inds]

        descs_norm[descs_norm > SIFT_THRES] = SIFT_THRES

        norms = np.sqrt(np.sum(descs_norm ** 2, axis=0))
        descs_norm /= norms

        descs[:, hcontrast_inds] = descs_norm

        return descs


def main():
    extractor = DenseSIFTExtractor(patch_size=32)
    I = lena()
    extractor.extract_descriptors(I)

if __name__ == '__main__':
    main()
