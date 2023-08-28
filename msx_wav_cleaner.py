#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Clean up MSX WAV files dumped from real cassettes.
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


def smooth_mean(wav_mean):
    '''
    Smooth local mean
    '''
    win = np.array(3 * [1])
    wav_mean_smoothed = ssignal.convolve(wav_mean, win, mode='same') / sum(win)
    return wav_mean_smoothed

def correct_mean(wav):
    '''
    Correct the wav by subtracting a local mean
    '''
    win = np.array(36 * [1]) # number of samples which cover a HIGH + LOW long pulse
    wav_mean = ssignal.convolve(wav, win, mode='same') / sum(win)
    wav_mean = smooth_mean(wav_mean)
    return wav - wav_mean, wav_mean



parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.parse_args()
args = parser.parse_args()

filename_input = args.input
filename_output = args.output

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

# Low peaks are around 1200 Hz
# High peaks are around 2400 Hz

f1 = 1200
f2 = 2 * f1

delta_L = 1100 # OK Sorcery and Rampart
delta_H = 1500

# Low pass
i = int(2 * len(wav) * (f1 - delta_L) / 44100)
D[:i] = 0

# High pass
i = int(2 * len(wav) * (f2 + delta_H) / 44100)
D[i:] = 0

R = idct(D, norm='ortho')
#save_wave(R, "R.wav")

wav = R.copy()

# Correct mean
wav, wav_mean = correct_mean(wav)
#save_wave(wav, "mean_corrected.wav")

# Saturate the signal at this moment.
# This is useful with tapes which contain wrongly too-long pulses, such as
# Sorcery at sample 4809431 (1:49.0570).
wav = np.clip(wav*8, a_min=-1, a_max=+1)

# Compute local mean
win = np.array(19 * [1]) # number of samples which cover a HIGH (or low) short pulse
wav_mean_abs = ssignal.convolve(np.abs(wav), win, mode='same') / sum(win)
#save_wave(wav_mean_abs, "wav_mean_abs.wav")

# Don't consider close-to-zero mean samples, since they'll
# blow up the mean correction which follows
low_energy_mean_indices = np.where(wav_mean_abs < 0.1) # 0.3 chosen with Rampart, 0.1 for Sorcery. ToDo: set this as a parameter
wav_mean_abs[low_energy_mean_indices] = 1.0

# Finally, remove noise in silences.
# Do do this, set to zero the samples in "wav" indexed by low_energy_mean_indices
# Note that the give the size of the mean kernel, the mean won't go to
# zero during the normal bit transitions of the wav
wav[low_energy_mean_indices] = 0.0

# Save result
save_wave(wav, filename_output)
