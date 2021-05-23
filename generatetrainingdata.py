from scipy.io import wavfile
import lab1_proto
import numpy as np
import os
import soundfile as sf
from librosa.feature import mfcc
rootdir = 'C:/Users/mikaelwestlund_fy17/school/speech/project'
#the label number. 0 = "backward", 1 = "bed", 2 = "bird"...
labelnumber = 0
#data is a list of dictionary where each dictionary contains the speach sample, sampling frequency, labelnumber and name
data = []
for root, dirs, files in os.walk(rootdir):
    for name in dirs:
        print(name)
        for place, dir, file in os.walk(rootdir+'/'+name):
            for f in file:
                intarray, sampfreq = sf.read(rootdir+'/'+name+'/'+f)
                if sampfreq != 16000:
                    print("SAMPFREQ NOT 16k")
                if len(intarray) == 16000:
                    mfccarray = np.transpose(mfcc(intarray, sr=sampfreq, n_mfcc=13, n_fft=512, hop_length=160, win_length=320))
                    #if row:
                    #    mfccarray = mfccarray.flatten()
                    ##elif column:
                    #    mfccarray = mfccarray.flatten(order='F')
                    data.append({'samplingfrequency': sampfreq, 'mfcc': mfccarray, 'label': labelnumber, 'utterance': name, 'filename': f})
                    

        labelnumber += 1

data = np.array(data)
np.save("data.npy", data)
