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
from scipy import signal as ssignal
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

# If it's stereo, subtract the noise/DC level from the signal
if len(wav.shape) == 2: # stereo
    if np.abs(wav[:, 0]).sum() > np.abs(wav[:, 1]).sum():
        wav = wav[:, 0] - wav[:, 1]
    else:
        wav = wav[:, 1] - wav[:, 0]

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

# Don't consider close-to-zero mean samples, since they'll
# blow up the mean correction which follows
low_energy_mean_indices = np.where(wav_mean_abs < 0.1)
wav_mean_abs[low_energy_mean_indices] = 1.0

# Finally, remove noise in silences.
# Do do this, set to zero the samples in "wav" indexed by low_energy_mean_indices
# Note that the give the size of the mean kernel, the mean won't go to
# zero during the normal bit transitions of the wav
wav[low_energy_mean_indices] = 0.0

# Save result
save_wave(wav, filename_output)
