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

def masterRunner(file,speedUpRate,outPut):
    
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
    #Setting Master X to model
    masterData = np.array([melVals,harmonyVals,dynaVals],dtype=object)
    X = masterData[:3].transpose()

    #Loading Model
    model_pkl_file = "music_model.pkl" 
    with open(model_pkl_file, 'rb') as modelFile:  
        model = pickle.load(modelFile)
    #Predictions
    ss_train = StandardScaler()
    X_master = ss_train.fit_transform(X)
    predictions = model.predict(X_master)
    print("Predictions made")
    #Setting list with predictions
    extPred = [0] * n
    i = 2
    while i < n-3:
        if predictions[i] and predictions[i+1]:
            extPred[i-2:i+3] = [1] * 5
        i += 1
    print("Creating new File")
    finalData = np.zeros(n,dtype = list)
    for i in range(n):
        if not extPred[i]:
            finalData[i] = pyrubberband.pyrb.time_stretch(data[int(i*frameLengthData):int((i+1)*frameLengthData)],sr = sr, rate = speedUpRate)
            #finalData[i] = librosa.effects.time_stretch(data[int(i*frameLengthData):int((i+1)*frameLengthData)], rate= 1.5)
        else:
            finalData[i] = data[int(i*frameLengthData):int((i+1)*frameLengthData)]

    finalFlatten = np.float32(np.concatenate([x.ravel() for x in finalData]))

    newName = file
    if '.' in outPut:
        newName = outPut
        if newName[-3:] != '.wav':
            newName = newName[:newName.index('.')] + ".wav"
    elif outPut != "":
        
        outputFile = newName.split("/")
        newName = outPut + "/" + outputFile[1]
        


    outputFile = newName.split(".")
    outputFile[0] += "spedUp"
    outputFile = outputFile[0] + "." + outputFile[1]

    scipy.io.wavfile.write(outputFile, sr, finalFlatten)

speedUpRate = float(input("Choose speed up rate: "))
file = input("Enter File/Folder Location ")
outPut = input("Enter new file location and name. (Empty input accepted): ")
if outPut != "" and '.' not in outPut:
    os.mkdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), outPut))
if '.' not in file:
    for elem in os.listdir(file):
        masterRunner(str(file + "/" + elem),speedUpRate,outPut)
        print("Done " + str(elem))
    print("Done all files")
else:
    masterRunner(file,speedUpRate,outPut)
    print("Done")
