# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:20:17 2023

@author: elean
"""
import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import soundfile as sf
import os
import matplotlib.pyplot as plt
import soundfile as sf


# Load audio files
#audio, fs = sf.read('9.wav', dtype='float32')


# Function to split audio into frames 
# Returns each frames length (in samples) and the indicie each frame starts
# at
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
    frames_start = pad_audio[indices.astype(np.int32, copy=False)]
    return frame_length, frames_start

# Function to compute the magnitude and phase spectra
def magnitude_phase_spectrum(frame_length, frames_start, nfft):
    # apply hamming window
    frames_start *= np.hamming(frame_length)
    # standard is 512 or 256
    # using rfft instead of fft to save time,rftt output half as long
    magnitude_spectrum = np.absolute(np.fft.rfft(frames_start, nfft)) 
    ####
    # plot magnitude spectra
    ###
    # power spectrum frames
    ###
    power_spectrum = ((1.0 / nfft) * ((magnitude_spectrum) ** 2))
    return magnitude_spectrum, power_spectrum

# Function to get the mel scale spectogram 
def mel_scale_filter_bank(channels, fs, nfft, power_frames):
    # find min and max in magnitude spectrum
    # convert into mel scale
    min_mel = 0
    max_mel = (2595 * np.log10(1 + (fs / 2) / 700))
    # generate evenly spaced points between the mel scale
    mel_points = np.linspace(min_mel, max_mel, channels + 2)
    print(len(mel_points))
    # convert mel point frequencies back to normal frequencies
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    # calculate centre frequencies
    centre_freq = np.floor((nfft + 1) * hz_points / fs)
    #  Convert these frequencies to indices of your magnitude spectrum
    # change variable names and add description
    fbank = np.zeros((channels, int(np.floor(nfft / 2 + 1))))   
    for m in range(1, channels + 1):
        f_m_minus = int(centre_freq[m - 1])   # left
        f_m = int(centre_freq[m])             # center
        f_m_plus = int(centre_freq[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - centre_freq[m - 1]) / (centre_freq[m] - centre_freq[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (centre_freq[m + 1] - k) / (centre_freq[m + 1] - centre_freq[m])
    fbanks = np.dot(power_frames, fbank.T)
    fbanks = np.where(fbanks == 0, np.finfo(float).eps, fbanks)
    #mel_spectogram = 20 * np.log10(fbanks)  

    #plt.imshow(mel_spectogram)
    return fbanks
### spectogram to mfcc
##apply to all audio files
# converts mel scale spectogram to mfcc
def mel_spectogram_mfcc(mel_spectogram, num_mfccs):
    # Step 1: Take the natural logarithm
    mfcc = np.log(mel_spectogram)  # Adding a small constant to avoid taking the log of zero
    #print(mfcc[0])
    # Step 2: Apply the DCT
    mfcc = dct(mfcc, type=2, axis=1, norm='ortho')[:, :num_mfccs]
    #plt.imshow(mfcc)
    return mfcc

#frame_length, frames_start = split_frames(0.2, 0.01, audio, fs)
#magnitude_spectra, phase_spectra = magnitude_phase_spectrum(frame_length, frames_start, 512)
#mel_spectogram = mel_scale_filter_bank(12, fs, 512, phase_spectra)
#mfcc = mel_spectogram_mfcc(mel_spectogram, 13)
#plt.imshow(mfcc)

folders = ['ben', 'charlie', 'chiedozie', 'carlos', 'el', 'ethan', 'francesca', 'jack', 'jake', 'james',
           'lindon', 'marc', 'nischal', 'robin', 'ryan', 'sam', 'seth', 'william', 'bonney', 'yubo']
for folder_name in folders:
    mfccs = []
    for file_index in range(20):
       
        file_name = f'{folder_name}/{file_index}.wav'  
        try:
            audio, fs = sf.read(file_name, dtype='float32')
            frame_length, frames_start = split_frames(0.2, 0.01, audio, fs)
            magnitude_spectra, phase_spectra = magnitude_phase_spectrum(frame_length, frames_start, 512)
            mel_spectogram = mel_scale_filter_bank(12, fs, 512, phase_spectra)
            mfcc = mel_spectogram_mfcc(mel_spectogram, 13)
            print(mfcc.shape)
            mfccs.append(mfcc)
            #np.save(f'{folder_name}/{file_index}.npy', mfccs)

        except Exception as e:
            print(f'Error processing {file_name}: {e}')
        #np.save(f'{folder_name}/{file_index}.npy', mfccs)
