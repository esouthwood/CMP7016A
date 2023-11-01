# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 21:56:03 2023

@author: elean
"""

# Load audio file
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import soundfile as sf
import os
import matplotlib.pyplot as plt
import soundfile as sf
audio, fs = sf.read('9.wav', dtype='float32')


t = np.arange(1/fs, 1/fs + len(audio)/fs, 1/fs)
def split_frames(length, overlap, audio, fs):
    # Frame length and overlap in samples
    frame_length = int(round(length * fs))
    frame_overlap = int(round(overlap * fs))

    audio_length = len(audio)

    # np.ceil rounds up to one to ensure there's at least one frame
    num_frames = int(np.ceil(float(np.abs(audio_length - frame_length)) / frame_overlap)) 
    # pad audio with zeros to ensure each frame is the same length
    pad_audio_length = num_frames * frame_overlap + frame_length
    z = np.zeros((pad_audio_length - audio_length))
    pad_audio = np.append(audio, z)
    # generate a matrix of indices that represent the positions of the start of each frame
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_overlap, frame_overlap),(frame_length, 1)).T
    print(indices[101][100])
    frames_start = pad_audio[indices.astype(np.int32, copy=False)]
    print(frames_start[101][100])
    return frame_length, frames_start

def magnitude_power_spectrum(frame_length, frames_start, nfft):
    # apply hamming window
    frames_start *= np.hamming(frame_length)
    # standard is 512 or 256
    # using rfft instead of fft to save time,rftt output half as long
    magnitude_spectrum = np.absolute(np.fft.rfft(frames_start, nfft))  
    # power spectrum frames
    power_spectrum = ((1.0 / nfft) * ((magnitude_spectrum) ** 2))
    return magnitude_spectrum, power_spectrum

def mel_scale_filter_bank(channels, fs, nfft, power_frames):
    #find min and max in magnitude spectrum
    # in mel scale turn into two steps
    min_mel = 0
    max_mel = (2595 * np.log10(1 + (fs / 2) / 700))
    # generate evvenly spaced points between the mel scale
    mel_points = np.linspace(min_mel, max_mel, channels + 2)
    print("length mel points ", len(mel_points))
    # conver mel frequencies back to normal frequencies
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    # calculate centre frequencies
    centre_freq = np.floor((nfft + 1) * hz_points / fs)
    #  Convert these frequencies to indices of your magnitude spectrum
    # change variable names and add description
    fbank = np.zeros((channels, int(np.floor(nfft / 2 + 1))))
    print(fbank.shape, fbank[0].shape)
    for m in range(1, channels + 1):
        f_m_minus = int(centre_freq[m - 1])   # left
        f_m = int(centre_freq[m])             # center
        f_m_plus = int(centre_freq[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - centre_freq[m - 1]) / (centre_freq[m] - centre_freq[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (centre_freq[m + 1] - k) / (centre_freq[m + 1] - centre_freq[m])
    # explain
    filter_banks = np.dot(power_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    # noise
    #filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    plt.imshow(filter_banks, origin='lower')
    return filter_banks
# n point fast fourier transform to get magnitude spectrum frames
nfft = 256
  
frame_length, frames_start = split_frames(0.2, 0.01, audio, fs)
magnitude, power = magnitude_power_spectrum(frame_length, frames_start, nfft)
mel_scale_fbank = mel_scale_filter_bank(12, fs, nfft, power)

import matplotlib.pyplot as plt
plt.imshow(mel_scale_fbank, origin='lower')

import cv2
mel_scale_fbank[10][11]
#dct
mfcc = dct(mel_scale_fbank, type=2, axis=1, norm='ortho')[:, 1 : (20 + 1)] # Keep 2-13

#noise
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
lift = 1 + (20 / 2) * np.sin(np.pi * n / 20)
mfcc *= lift

#plt.imshow(mfcc, origin='lower')
