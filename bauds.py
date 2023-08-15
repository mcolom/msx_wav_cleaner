#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Obtain the bauds locally.
This tool is mainly to help the TSX preservation group.
Copyright (c) 2023 mcolom
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

import argparse
import numpy as np
from scipy.fftpack import dct, idct
from scipy import signal as ssignal
from scipy.io import wavfile
from lib import read_wave, save_wave
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.parse_args()
args = parser.parse_args()

filename_input = args.input
filename_output = args.output

half_width = 30000
th = 6 # Threshold for absolute DCT values
f1 = 1200
f2 = 2 * f1
delta = 200 # This actually depends on half_width



plt.ion()
plt.close("all")

# Read input signal
wav = read_wave(filename_input)

# If it's stereo, subtract the noise/DC level from the signal,
# Check also if indeed one of the channels has significantly less energy than the other.
if len(wav.shape) == 2: # stereo
    if np.abs(wav[:, 0]).sum() > np.abs(wav[:, 1]).sum():
        wav = wav[:, 0]
    else:
        wav = wav[:, 1]

# Apply low and high pass filters
D = dct(wav, norm='ortho')

d = 1100

# Low pass
i = int(2 * len(wav) * (f1 - d) / 44100)
D[:i] = 0

# High pass
i = int(2 * len(wav) * (f2 + d) / 44100)
D[i:] = 0

wav = idct(D, norm='ortho')


# test
wav = wav[44100*40:44100*45]


R = np.zeros(len(wav) - 2*half_width)

for i in tqdm(np.arange(half_width, len(wav) - half_width)):
    S = wav[i : i + half_width + 1]
    
    # D is the absolute DCT-II
    D = np.abs(dct(S, norm='ortho'))
    D[0] = 0 # Kill DC

    # Indices of f1 and f2 in the DCT
    #idx_f1 = int(np.round(f1 * len(D) / 22050))
    idx_f2 = int(np.round(f2 * len(D) / 22050))

    """
    plt.figure()
    plt.plot(np.abs(D[1:]))
    plt.show()

    plt.figure()
    plt.plot(S)
    plt.show()
    """

    #D1 = D[idx_f1 - delta: idx_f1 + delta]
    #m1 = np.argmax(D1)
    #
    D2 = D[idx_f2 - delta: idx_f2 + delta]
    m2 =  np.argmax(D2)

    #r1 = (m1 + idx_f1 - delta) / len(D) * 22050 if D1[m1] >= th else None
    r2 = (m2 + idx_f2 - delta) / len(D) * 22050 if D2[m2] >= th else None

    """
    if r1 is not None and r2 is not None:
        r = r2
    elif r2 is not None:
        r = r2
    elif r1 is not None:
        r = 2 * r1
    else:
        r = None
    """
    r = r2

    R[i - half_width] = r

plt.figure()
plt.plot(R)
plt.show()

# Soften with the mean
win = np.array(201 * [1])
R[R is None] = 2400
R[np.isnan(R)] = 2400
Rf = ssignal.convolve(R, win, mode='same') / sum(win)

plt.figure()
plt.plot(Rf[half_width:len(Rf)-half_width])
plt.show()


plt.savefig(f"result_half_width_{half_width}_delta_{delta}.png")
