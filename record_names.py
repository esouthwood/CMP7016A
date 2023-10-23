# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:02:14 2023

@author: elean
"""

import sounddevice as sd
import soundfile as sf
import os

# list of names 
# could read it in from text file to make it more effiecient
names = ['ben', 'charlie', 'chiedozie', 'carlos',
         'el', 'ethan', 'francesca', 'jack', 'jake',
         'james', 'lindon', 'marc', 'nischal', 'robin',
         'ryan', 'sam', 'seth', 'william', 'bonney',
         'yubo']

# set frequency and seconds
fs = 16000
seconds = 3
audio_names = []

# record each name 20 times
for name in names:
    count = 0
    for i in range(20):
        # create a directory for each name if one doesn't
        # already exist
        directory = name+'/'
        directoryExists = os.path.exists(directory)
        if not directoryExists:
            os.mkdir(directory)

        # so the person recording knows what name to say
        print(name,count)
        
        r = sd.rec(seconds * fs, samplerate=fs, channels=1)
        sd.wait()
        
        # save the audio file to directory
        path = directory+str(count)+'.wav'
        sf.write(path, r, fs)
        count+=1
        
        audio_names.append(r)
        
