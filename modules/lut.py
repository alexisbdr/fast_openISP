# File: lut.py
# Description: 3D LUT
# Created: 2022/08/17 
# Author: Alexis Baudron (alexis.baudron@sensebrain.site)


import numpy as np
from colour import LUT3D

from .basic_module import BasicModule, register_dependent_modules
from .helpers import pad, shift_array, mean_filter


class LUT(BasicModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        