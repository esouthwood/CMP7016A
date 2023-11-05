import numpy as np
from scipy.io import wavfile
import soundfile as sf

# Reading the noise file and get its data and sampling rate
noiseFile, fs = sf.read('noise.wav', dtype='float32')

# List of folders containing speech files
folders = ['ben', 'charlie', 'chiedozie', 'carlos', 'el', 'ethan', 'francesca', 'jack', 'jake', 'james',
           'lindon', 'marc', 'nischal', 'robin', 'ryan', 'sam', 'seth', 'william', 'bonney', 'yubo']


for folder_name in folders:
    
    for file_index in range(3):
        # Constructing the file path for the current speech file
        file_name = f'{folder_name}/{file_index}.wav'
        
        try:
            # Reading the speech file and get its data and sampling rate
            speechFile, fs_speech = sf.read(file_name, dtype='float32')
            
            # Ensuring both speech and noise signals have the same length
            min_length = min(len(speechFile), len(noiseFile))
            speechFile = speechFile[:min_length]
            noiseFile = noiseFile[:min_length]

            # Adding noise to speech to create noisy speech
            noisy_speech = speechFile + noiseFile
            
            # Specifing the desired filename for the noisy speech file including folder name
            noisy_speech_file = f'{folder_name}/{file_index}_noisy_speech.wav'
            
            # Saving the noisy speech to a new WAV file
            wavfile.write(noisy_speech_file, fs_speech, noisy_speech)
        
        except Exception as e:
           
            print(f'Error processing {file_name}: {e}')

