import numpy as np
import matplotlib.pyplot as plt
from python_speech_features import mfcc
from python_speech_features import logfbank
import lab1_proto



def loaddata():
    return np.load("data.npy", allow_pickle=True)

#compute mean and standard deviation of the training set
def computeMeanSTD(trainingdata):

    #calculate mean
    sum = 0
    for i in range(len(trainingdata)):
        sum += trainingdata[i]['mfcc']
    mean = sum/len(trainingdata)

    #calculate variance
    sum = 0
    for i in range(0, len(trainingdata)):
        sum += np.square(trainingdata[i]['mfcc']-mean)
    std = np.sqrt(sum/len(trainingdata))

    return mean, std

#normalize the sets to the mean and standard deviation and also flattens the mfcc
def normalization(trainingset, validationset, testingset, mean, std):
    returnlist = []
    samplelist, labellist = np.zeros((len(trainingset), np.shape(trainingset[0]['mfcc'])[0]*np.shape(trainingset[0]['mfcc'])[1])), np.zeros(len(trainingset))
    for i in range(0, len(trainingset)):
        trainingset[i]['mfcc'] = ((trainingset[i]['mfcc'] - mean)/std).flatten()
        samplelist[i, :] = trainingset[i]['mfcc']
        labellist[i] = trainingset[i]['label']
    returnlist.append((samplelist, labellist))

    samplelist, labellist = np.zeros((len(validationset), np.shape(validationset[0]['mfcc'])[0]*np.shape(validationset[0]['mfcc'])[1])), np.zeros(len(validationset))
    for i in range(0, len(validationset)):
        validationset[i]['mfcc'] = ((validationset[i]['mfcc']- mean)/std).flatten()
        samplelist[i, :] = validationset[i]['mfcc']
        labellist[i] = validationset[i]['label']
    returnlist.append((samplelist, labellist))

    samplelist, labellist = np.zeros((len(testingset), np.shape(testingset[0]['mfcc'])[0]*np.shape(testingset[0]['mfcc'])[1])), np.zeros(len(testingset))
    for i in range(0, len(testingset)):
        testingset[i]['mfcc'] = ((testingset[i]['mfcc']- mean)/std).flatten()
        samplelist[i, :] = testingset[i]['mfcc']
        labellist[i] = testingset[i]['label']
    returnlist.append((samplelist, labellist))

    return returnlist

#de-normalize (scale up) the generated sample with the mean and standarddeviation
#also reshapes the flattened sample in to (dim_1, dim_2) sized matrix
def denormalization(sample, mean, std, dim_1, dim_2):
    sample = np.reshape(sample, (dim_1, dim_2))
    return sample*std + mean

#split the data into training, validation and testing set according to the percentages.
def splitset(percenttraining, percentvalidation, percenttest, data):
    if percenttraining + percentvalidation + percenttest != 1.0:
        print("probabilities must sum to 1")
        return None
    else:
        utterance_dict = {}
        
        for utterance in data:
            label = utterance["label"]
            if label in utterance_dict:
                utterance_dict[label].append(utterance)
            else:
                utterance_dict[label] = [utterance]
        trainingset = []
        validationset = []
        testset = []
        for label in utterance_dict.keys():
            np.random.shuffle(utterance_dict[label])
            numberofdatasamples = len(utterance_dict[label])
            trainingset += utterance_dict[label][0:int(percenttraining*numberofdatasamples)]
            validationset += utterance_dict[label][int(percenttraining*numberofdatasamples): int(percenttraining*numberofdatasamples) + int(percentvalidation*numberofdatasamples)]
            testset += utterance_dict[label][int(percenttraining*numberofdatasamples) + int(percentvalidation*numberofdatasamples): numberofdatasamples]

        np.random.shuffle(trainingset)
        np.random.shuffle(validationset)
        np.random.shuffle(testset)


        print("trainingset", len(trainingset))
        print("validationset", len(validationset))
        print("testset", len(testset))

        return trainingset, validationset, testset

def labeltoname1digit(labelnumber):
    if labelnumber == 0 or labelnumber =='0':
        return "backward"
    elif labelnumber == 1 or labelnumber =='1':
        return "bed"
    elif labelnumber == 2 or labelnumber =='2':
        return "bird"
    elif labelnumber == 3 or labelnumber =='3':
        return "cat"
    elif labelnumber == 4 or labelnumber =='4':
        return "dog"
    elif labelnumber == 5 or labelnumber =='5':
        return "down"
    elif labelnumber == 6 or labelnumber =='6':
        return "eight"
    elif labelnumber == 7 or labelnumber =='7':
        return "five"
    elif labelnumber == 8 or labelnumber =='8':
        return "follow"
    elif labelnumber == 9 or labelnumber =='9':
        return "forward"
    elif labelnumber == 10 or labelnumber =='10':
        return "four"
    elif labelnumber == 11 or labelnumber =='11':
        return "go"
    elif labelnumber == 12 or labelnumber =='12':
        return "happy"
    elif labelnumber == 13 or labelnumber =='13':
        return "house"
    elif labelnumber == 14 or labelnumber =='14':
        return "learn"
    elif labelnumber == 15 or labelnumber =='15':
        return "left"
    elif labelnumber == 16 or labelnumber =='16':
        return "marvin"
    elif labelnumber == 17 or labelnumber =='17':
        return "nine"
    elif labelnumber == 18 or labelnumber =='18':
        return "no"
    elif labelnumber == 19 or labelnumber =='19':
        return "off"
    elif labelnumber == 20 or labelnumber =='20':
        return "on"
    elif labelnumber == 21 or labelnumber =='21':
        return "one"
    elif labelnumber == 22 or labelnumber =='22':
        return "right"
    elif labelnumber == 23 or labelnumber =='23':
        return "seven"
    elif labelnumber == 24 or labelnumber =='24':
        return "sheila"
    elif labelnumber == 25 or labelnumber =='25':
        return "six"
    elif labelnumber == 26 or labelnumber =='26':
        return "stop"
    elif labelnumber == 27 or labelnumber =='27':
        return "three"
    elif labelnumber == 28 or labelnumber =='28':
        return "tree"
    elif labelnumber == 29 or labelnumber =='29':
        return "two"
    elif labelnumber == 30 or labelnumber =='30':
        return "up"
    elif labelnumber == 31 or labelnumber =='31':
        return "visual"
    elif labelnumber == 32 or labelnumber =='32':
        return "wow"
    elif labelnumber == 33 or labelnumber =='33':
        return "yes"
    elif labelnumber == 34 or labelnumber =='34':
        return "zero"
    else:
        return None

def labeltoname(labelnumber):
    if type(labelnumber) == int:
        return labeltoname1digit(labelnumber)

    elif type(labelnumber) == list or type(labelnumber) == np.ndarray:
        output = []
        for i in range(0, len(labelnumber)):
            output.append(labeltoname1digit(labelnumber[i]))
        return output
    else:
        return None


def lengthHistogram(data):
    """
    Input: data
    Output: listoflengths, dictionary with keys = length of samples that are not 16000,
    values = number of occurances of this key in the data.
    Plots a histogram of all the different lengths of our data objects and
    how many of each length there are. Also returns this dictionary."""
    listoflengths = {}
    hasprintedlarge = False
    hasprintedsmall = False
    for dataobject in data:
        if len(dataobject["sample"]) in listoflengths:
            pass
        else:
            if str(len(dataobject["sample"])) in listoflengths:
                listoflengths[str(len(dataobject["sample"]))] = listoflengths[str(len(dataobject["sample"]))] +1
            else:
                listoflengths[str(len(dataobject["sample"]))] = 1
    plt.bar(listoflengths.keys(), listoflengths.values())
    plt.title("The length of the data files")
    plt.ylabel("Number of files")
    plt.xlabel("Lengths of samples")
    plt.show()
    return listoflengths

def shortClasses(data):
    """
    Input: data
    Output: classDict, dictionary with keys = labels, values = number oof samples with
    less than 16000 length .

    Description: Plots a histogram containing all the data samples with less than 16000 length
    for each class (label). Also returns this dictionary."""
    classDict = {}
    for dataobject in data:
        if len(dataobject["sample"]) == 16000:
            pass
        else:
            if str(dataobject["label"]) in classDict:
                classDict[str(dataobject["label"])]  = classDict[str(dataobject["label"])] +1
            else:
                classDict[str(dataobject["label"])] = 1
    a = []
    for i in classDict.keys():
        a.append(i)
    b = []
    for i in classDict.values():
        b.append(i)

    plt.bar(labeltoname(a), b,)
    plt.title("The classes with smaller than 16000 length")
    plt.ylabel("Number of files")
    plt.xticks(rotation=45)
    plt.xlabel("Utterance")
    plt.show()
    return classDict


"""
print("loading data")
data = loaddata()
t, v, test = splitset(0.9, 0.05, 0.05, data)
label_counter = {}
for d in test:
    if d["label"] in label_counter:
        label_counter[d["label"]] += 1 
    else:
        label_counter[d["label"]] = 0 
print(label_counter)
print("loading complete")
print(len(data))
print(data[-1]['label'])

minvalue = np.inf
maxvalue = -np.inf
for i in range(0, len(data)):
    minv = np.min(data[i]["mfcc"])
    maxv = np.max(data[i]["mfcc"])
    if minv<minvalue:
        minvalue = minv
    elif maxv>maxvalue:
        maxvalue = maxv
print(minvalue)
print(maxvalue)
"""