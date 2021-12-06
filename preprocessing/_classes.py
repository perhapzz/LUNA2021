import os
import numpy as np
from glob import glob
import scipy.misc
import SimpleITK as sitk
from skimage.measure import regionprops

from configs import RESOURCES_PATH, OUTPUT_PATH
from preprocessing.utility import get_segmented_lungs


class CTScan(object):
    def __init__(self, seriesuid):
        self._seriesuid = seriesuid
        paths = glob(f'''{RESOURCES_PATH}/{self._seriesuid}/*.mhd''')
        path = paths[0]
        self._ds = sitk.ReadImage(path)
        self._spacing = np.array(list(reversed(self._ds.GetSpacing())))
        self._origin = np.array(list(reversed(self._ds.GetOrigin())))
        self._image = sitk.GetArrayFromImage(self._ds)
        self._mask = None

    def preprocess(self):
        self._resample()
        self._segment_lung_from_ct_scan()
        self._normalize()
        # self._zero_center()
        # self._change_coords()

    def save_preprocessed_image(self):
        file_path = f'{self._seriesuid}/{self._seriesuid}_clean.npy'
        os.makedirs(f'{OUTPUT_PATH}/{self._seriesuid}', exist_ok=True)
        np.save(f'{OUTPUT_PATH}/{file_path}', np.expand_dims(self._image, axis= 0))

    def get_info_dict(self):
        (min_z, min_y, min_x, max_z, max_y, max_x) = (None, None, None, None, None, None)
        for region in regionprops(self._mask):
            min_z, min_y, min_x, max_z, max_y, max_x = region.bbox
        assert (min_z, min_y, min_x, max_z, max_y, max_x) != (None, None, None, None, None, None)
        min_point = (min_z, min_y, min_x)
        max_point = (max_z, max_y, max_x)
        return {'seriesuid': self._seriesuid, 'radii': self._radii, 'centers': self._centers,
                'spacing': list(self._spacing), 'lungs_bounding_box': [min_point, max_point], 'class': self._clazz}

    def _resample(self):
        spacing = np.array(self._spacing, dtype=np.float32)
        new_spacing = [1, 1, 1]
        imgs = self._image
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = scipy.ndimage.interpolation.zoom(imgs, resize_factor, mode='nearest')
        self._image = imgs
        self._spacing = true_spacing

    def _segment_lung_from_ct_scan(self):
        result_img = []
        result_mask = []
        for slicee in self._image:
            rimg, rmsk = get_segmented_lungs(slicee)
            result_img.append(rimg)
            result_mask.append(rmsk)
        self._image = np.asarray(result_img)
        self._mask = np.asarray(result_mask, dtype=int)

    def _normalize(self):
        MIN_BOUND = -1200
        MAX_BOUND = 600.
        self._image = (self._image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        self._image[self._image > 1] = 1.
        self._image[self._image < 0] = 0.
        self._image *= 255.

    def _zero_center(self):
        PIXEL_MEAN = 0.25 * 256
        self._image = self._image - PIXEL_MEAN
