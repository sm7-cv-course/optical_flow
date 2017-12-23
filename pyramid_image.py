import numpy as np
import cv2

# Class PyramidImage.
#
class PyramidImage:
    def __init__(self, img):
        # Difference between layers (downscale).
        self.__scale = 1.0
        # Number of pyramidal layers.
        self.__pyr_num = 1
        # 'decimation', 'nearest', 'bicubic',
        self.__interp_type = 'decimation'
        self.__orig = img
        # key - scale, value - 1 or 3 layers.
        # self.__layers_dict[]

    # Build pyramid images.
    def build_ovr(self, n_ovr, scale):
        self.__pyr_num = n_ovr
        self.__scale = scale
        cur_img = self.__orig
        cur_scale = 1
        for k in range(self.__pyr_num):
            cur_scale = cur_scale * self.__scale
            if self.__interp_type == 'nearest':
                img_dwnsmpl = cv2.resize(cur_img, None, fx=self.__scale, fy=self.__scale, interpolation=cv2.INTER_NEAREST)
                self.__layers_dict[cur_scale] = img_dwnsmpl
            if self.__interp_type == 'cubic':
                img_dwnsmpl = cv2.resize(cur_img, None, fx=self.__scale, fy=self.__scale, interpolation=cv2.INTER_CUBIC)
                self.__layers_dict[cur_scale] = img_dwnsmpl

    # Get layer at index in pyramid vector.
    def get_layer_at_index(self, index):
        return self.__layers_vec[index]

    def get_layer(self, scale):
        """
        Get layer with closest to given scale.
        """
        #for s in range(self.__pyr_num):

    def get_orig_image(self):
        return self.__orig
