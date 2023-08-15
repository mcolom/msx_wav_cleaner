#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Support functions
"""

#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.io import wavfile

def read_wave(filename):
    '''
    Read a WAV file
    '''
    samplerate, wavdata = wavfile.read(filename)

    assert samplerate == 44100
    if wavdata.dtype == np.uint8:
        wavdata = (wavdata.astype(int) - 128) / 128
    elif wavdata.dtype == np.uint16:
        wavdata = (wavdata.astype(int) - 32768) / 32768
    elif wavdata.dtype == np.int8:
        wavdata = wavdata.astype(int) / 128
    elif wavdata.dtype == np.int16:
        wavdata = wavdata.astype(int) / 32768
    else:
        raise ValueError(wavdata.dtype)

    return wavdata

def save_wave(wavdata, filename):
    '''
    Write a WAV file
    '''
    wavdata = np.clip(32768*wavdata, a_min=-32768, a_max=32767)
    wavfile.write(filename, 44100, wavdata.round().astype(np.int16))
