import numpy as np

import albumentations as A
from mmcv.runner import obj_from_dict
from . import transforms


class ExtraAugmentation(object):

    def __init__(self, **kwargs):
        self.transform = self.transform_from_dict(**kwargs)

    def transform_from_dict(self, **kwargs):
        if 'transforms' in kwargs:
            kwargs['transforms'] = [self.transform_from_dict(**transform) for transform in kwargs['transforms']]
        try:
            return obj_from_dict(kwargs, transforms)
        except AttributeError:
            return obj_from_dict(kwargs, A)

    def __call__(self, img):
        data = self.transform(
            image=img,
        )
        return data['image']
