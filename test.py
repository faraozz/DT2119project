from librosa.feature import mfcc
import librosa
from librosa.feature.inverse import mfcc_to_audio
import numpy as np
from scipy.io import wavfile
import IPython.display as ipd
import soundfile as sf

data, rate = sf.read("3o33951a.wav")


outputbefore = sf.write("output1.wav", data, rate)
mfccs = mfcc(data)

reconstruction = mfcc_to_audio(mfccs)
outputafter = sf.write("output2.wav", reconstruction, rate)
