

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:20:17 2023

@author: elean
"""
import numpy as np
from scipy.fftpack import dct
import soundfile as sf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, InputLayer
from tensorflow.keras.optimizers import Adam


# Function to split audio into frames 
# Returns each frames length (in samples) and the indicie each frame starts
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
    mel_spectogram = 20 * np.log10(fbanks)  

    #plt.imshow(mel_spectogram)
    return fbanks, mel_spectogram
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
            mel_spectogram, ms = mel_scale_filter_bank(18, fs, 512, phase_spectra)
            mfcc = mel_spectogram_mfcc(mel_spectogram, 18)
            mfccs.append(mfcc)
            np.save(f'{folder_name}/{file_index}.npy', mfccs)

        except Exception as e:
            print(f'Error processing {file_name}: {e}')
        #np.save(f'{folder_name}/{file_index}.npy', mfccs)
mfccs



data=[]
labels2=[]

for folder_name in folders:
  
    for file_index in range(20):
        file_name = f'{folder_name}/{file_index}.wav'  
      
        try:
          
            #print(file_name)
            speechFile, fs = sf.read(file_name, dtype='float32')
            
            
            mfcc_file = np.load(f'{folder_name}/{file_index}.npy')
            mfcc_file = mfcc_file[0, :, :]

            data.append(mfcc_file)
            labels2.append(folder_name)
            
        except Exception as e:
          
            print(f'Error processing {file_name}: {e}')
            

data = np.array(data)   
data.shape
LE=LabelEncoder()
labels=to_categorical(LE.fit_transform(labels2))

X_train, X_tmp, y_train, y_tmp = train_test_split(data, 
labels, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,
test_size=0.5, random_state=0)

X_train.shape
# reshape
def create_model():
    numClasses=20
    model=Sequential()
    model.add(InputLayer(input_shape=(280, 18, 1)))
    #32 not 16
    model.add(Conv2D(24, (3, 3), activation='relu'))
    model.add(Flatten())
    #100 not 50
    model.add(Dense(100, activation='relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model

model = create_model()
model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
model.summary()

num_epochs = 15
num_batch_size = 64


# Train the model with callbacks
history = model.fit(X_train, y_train, validation_data=(X_val,y_val), batch_size=num_batch_size, epochs=num_epochs,verbose=1)

# Save the best model weights
#model.load_weights('best_model.h5')
model.save_weights('digit_classification.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

from sklearn import metrics

predicted_probs=model.predict(X_test,verbose=0)
predicted=np.argmax(predicted_probs,axis=1)
actual=np.argmax(y_test,axis=1)
accuracy = metrics.accuracy_score(actual, predicted)
print(f'Accuracy: {accuracy * 100}%')

confusion_matrix = metrics.confusion_matrix(np.argmax(y_test,axis=1), predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix =confusion_matrix)
cm_display.plot()

testing_names = ['sam', 'marc', 'bonney', 'chiedozie', 'william', 'jack', 'francesca',
                 'el', 'james', 'charlie']

testing_mfccs = []
for name in testing_names:
    file_name = f'testing/{name}.wav'  

    audio, fs = sf.read(file_name, dtype='float32')
    frame_length, frames_start = split_frames(0.2, 0.01, audio, fs)
    magnitude_spectra, phase_spectra = magnitude_phase_spectrum(frame_length, frames_start, 512)
    mel_spectogram, ms = mel_scale_filter_bank(18, fs, 512, phase_spectra)
    mfcc = mel_spectogram_mfcc(mel_spectogram, 18)
    #print(mfcc.shape)
    testing_mfccs.append(mfcc)
testing_mfccs = np.array(testing_mfccs)
testing_mfccs.shape
X_train.shape
predicted_probs2=model.predict(testing_mfccs,verbose=0)
predicted2=np.argmax(predicted_probs2,axis=1)
#actual=np.argmax(y_test,axis=1)
results = []
for i in predicted2:
    results.append(folders[i])
    
results
matching_elements = sum(1 for x, y in zip(testing_names, results) if x == y)

final_accuracy = (matching_elements // len(results)) * 100
print(results)
print(final_accuracy)
