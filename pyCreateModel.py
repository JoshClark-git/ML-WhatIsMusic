import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import soundfile
import os
import pickle
import librosa
import librosa.display
import librosa.display as lplt
from IPython.display import Audio
import pyrubberband
from pydub import AudioSegment
import time
import statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle
from scipy.io import wavfile

file = input("Enter File Location: ")
supervised = input("Enter Supervised File Location")
fileFormat = file.split('.')[-1]
print("Loading: " + file)
if fileFormat.upper() not in soundfile.available_formats():
    newFile = file.split('.')[0] + ".wav"
    sound = AudioSegment.from_mp3(file)
    sound.export(newFile, format="wav")
    file = newFile
data, sr = librosa.load(file)
print("Successfully loaded: ",file)

#Master Characteristics
time = int(librosa.get_duration(y=data, sr=sr))
frameLengthData = len(data)/int(librosa.get_duration(y=data, sr=sr))
n = (int(librosa.get_duration(y=data, sr=sr)))
step = len(data)/n
print("Reading Melody")
#melody
chroma = librosa.feature.chroma_stft(y=data,sr=sr)
lenChroma = len(chroma[0])
frameLengthChroma = len(chroma[0])/n
melVal = n * [0]
nthPercent = 0.1
currPercent = int(nthPercent*n)
for i in range(n):  
    stepper = int(i*frameLengthChroma)
    for j in range(12):
        for k in range(int(frameLengthChroma)-5):
            if all(chroma[j][stepper+k:stepper+k+5] == 1):
                melVal[i] += 1
print("Read Melody")
#dynamics
print("Reading Dynamics")
dynaVal = [0] * n
for i in range(n):
    dynaVal[i] = sum(abs(data[int(i*frameLengthData):int((i+1)*frameLengthData)]))
print("Read Dynamics")
#harmony
print("Reading Harmony")
y = librosa.effects.harmonic(data)
tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
transTonnetz = tonnetz.transpose()
frameLengthTonnetz = len(tonnetz[0])/n
harmonyVal = [0] * n
nthPercent = 0.1
currPercent = int(nthPercent*n)
for i in range(n):
    best = -1
    for j in range(int(frameLengthTonnetz)):
        best = max(best,np.amax(transTonnetz[int(i*frameLengthTonnetz)+j]) - np.amin(transTonnetz[int(i*frameLengthTonnetz)+j]))
    harmonyVal[i] = best
print("Read Harmony")

#Setting List Ranges
melVals = [0] * n
harmonyVals = [0] * n
dynaVals = [0] * n
for i in range(2,n-3):
    melVals[i] = statistics.mean(melVal[i-2:i+3])
    harmonyVals[i] = statistics.mean(harmonyVal[i-2:i+3])
    dynaVals[i] = statistics.mean(dynaVal[i-2:i+3])

#target Setup
with open(supervised) as f:
    lines = f.readlines()
target = [0] * n
for line in lines:
    intLine = line.split("-")
    if(len(intLine) == 1):
        continue
    intLine[0] = int(intLine[0])
    intLine[1] = int(intLine[1][:-1])
    for i in range(intLine[0],intLine[1]):
        target[i] = 1

masterData = np.array([melVals,harmonyVals,dynaVals,target],dtype=object)
X = masterData[:3].transpose()
y = masterData[3]
y=y.astype('int')

#Modeling
#Final Prediction
#Final Model
ss_train = StandardScaler()
X_master = ss_train.fit_transform(X)
y_master = y
newModel = KNeighborsClassifier()
newModel.fit(X_master, y_master)

# save the classification model as a pickle file
model_pkl_file = "music_model.pkl"  

with open(model_pkl_file, 'wb') as file:  
    pickle.dump(newModel, file)
