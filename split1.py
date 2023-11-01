import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pathlib import Path
import os
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


folders = ['ben', 'charlie', 'chiedozie', 'carlos', 'el', 'ethan', 'francesca', 'jack', 'jake', 'james',
           'lindon', 'marc', 'nischal', 'robin', 'ryan', 'sam', 'seth', 'william', 'bonney', 'yubo']
data=[]
labels=[]

for folder_name in folders:
  
    for file_index in range(20):
        file_name = f'{folder_name}/{file_index}.wav'  
      
        try:
          
            print(file_name)
            speechFile, fs = sf.read(file_name, dtype='float32')
            
            
            mfcc_file = np.load(f'{folder_name}/{file_index}.npy')
                                   
            data.append(mfcc_file)
            labels.append(folder_name)
    
            print(f'Folder: {folder_name}, File: {file_index}, Shape: {speechFile.shape}, Sampling Frequency: {fs}')
        
        except Exception as e:
          
            print(f'Error processing {file_name}: {e}')
            
            
labels = np.array(labels)
data = np.array(data)   
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
labels=to_categorical(LE.fit_transform(labels))

from sklearn.model_selection import train_test_split
X_train, X_tmp, y_train, y_tmp = train_test_split(data, 
labels, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp,
test_size=0.5, random_state=0)
        
            

