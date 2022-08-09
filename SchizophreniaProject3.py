import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

import pandasql as ps




#dataframes = ["Study_A.csv", "Study_B.csv", "Study_C.csv", "Study_D.csv"]
dataframes = [ "Study_B.csv","Study_C.csv"]
#dataframes = ["Study_D.csv"]

testData = ["Study_E.csv"]

trains = []
for x in dataframes:
    trains.append( pd.read_csv(x))
train = pd.concat(trains)

test = pd.read_csv(testData[0])

#days in 126 days
# we're looking for consistent data in the 190-210 day range

testData = test.to_numpy()
trainData = train.to_numpy()

def featureExtractor(inpTens, testSet=False):
    final = []
    if not testSet:
        final.append(inpTens[7:-1])
    else:
        final.append(inpTens[7:])
    return np.array(final)

def sequencify(inp, test = 0):
    patient = inp[0][2]
    sequences = []
    lens = []
    patients = []
    leng = 1
    runningAggregation = featureExtractor(inp[0], test)

    for x in inp[1:]:
        if(x[2] == patient):
            leng += 1
            runningAggregation = np.vstack((runningAggregation, featureExtractor(x, test)))
        else:
            sequences.append(runningAggregation)
            patients.append(patient)
            lens.append(leng)
            patient = x[2]
            leng = 1
            runningAggregation = featureExtractor(x, test)
    sequences.append(runningAggregation)
    patients.append(patient)
    lens.append(leng)
    return sequences, patients


preProcessed, _ = sequencify(trainData)
preTrained, patientLabels = sequencify(testData, True)
    


#we need to remove duplicate elements in the preTrained
def getUniqueTimePoints(inp):
    toReturn = []
    for x in inp:
        _, inds = np.unique(x[:,0].astype(int), return_index=True)
        toReturn.append(x[inds])
    return toReturn

def getProcessedTimeSeries(inp):
    series = []
    for inputs in inp:
        moddedSeries = []
        valueRanges = [35,77,119,161,203]
        tolerance = 15
        index = 1
        if(inputs[0][0] == 0 and len(inputs) > 1):
            moddedSeries.append(inputs[0])
            while len(valueRanges) != 0:
                if(inputs[index,0] >= valueRanges[0] - tolerance and inputs[index,0] <= valueRanges[0] + tolerance):
                    #print("peepee")
                    #val = inputs[index,0]
                    moddedSeries.append(inputs[index])
                    valueRanges.pop(0)
                    #if(len(valueRanges) != 0):
                    #    valueRanges[0] += val
                index += 1
                if(index >= len(inputs)):
                    break
            #if(len(moddedSeries) >= 2):
                #print(moddedSeries)
            if(len(moddedSeries) == 6):
                series.append(np.vstack(moddedSeries))
    return series  
    
preTrained = getUniqueTimePoints(preTrained)
preProcessed = getUniqueTimePoints(preProcessed)
processed = getProcessedTimeSeries(preProcessed)



X = [x[:-1] for x in processed]
y = [x[-1,-1] for x in processed]
    

def doubleExponentialMean(inp, a = 0.5, gamma = 0.8):
    predictions = []
    for x in inp:
        data = x.copy()
        pred = data[0,-1]
        grad = data[1,-1] - data[0,-1]
        for x in range(1, len(data)):
            prev = pred
            pred = a*data[x,-1] + (1-a) * (pred + grad)
            grad = gamma*(pred - prev) + (1-gamma)*grad
        predictions.append(pred)
    return predictions

def getVals(inp):
    liste = []
    for x in range(inp):
        liste.append(x/inp)
    return liste
aVals = getVals(10)
gammaVals = getVals(10)
besta = 0
bestgamma = 0
bestVal = 1000000
for a in aVals:
    for gamma in gammaVals:
        yhat = doubleExponentialMean(X, a, gamma)
        yhat = np.array(yhat)
        y = np.array(y)
        acq = np.average((y-yhat)**2)**(0.5)
        if(acq < bestVal):
            besta = a
            bestgamma = gamma
            bestVal = acq
print(besta)
print(bestgamma)
print(bestVal)

besta = 0.875
bestgamma = 0.325

#predictions 
#processed actually has the time series data we're interested in using
#173 datapoints using the training data

viewPretrained = [x for x in preTrained if len(x) < 4 and len(x) > 1]
ShortInds = []
LongInds = []
GoingtoModelInds = []
modelData = []
finalPANS = []
for x in range(len(preTrained)):
    if len(preTrained[x]) < 5:
        ShortInds.append(x)
        finalPANS.append(preTrained[x][-1,-1])
    else: 
        LongInds.append(x)
        if(preTrained[x][-1,0] >= 150 and preTrained[x][-2,0] >= 110):
            GoingtoModelInds.append(x)
            modelData.append(preTrained[x])

predictions = doubleExponentialMean(modelData, besta, bestgamma)

#we're trying neural networks now

    
        


y = []
X = []
preProcessed = [x for x in preProcessed if (x.ndim != 0 and x.shape[0] >= 2) ]

def generateInputs(inps, endIndex = -2):
    X = []
    for x in inps:
        if(x.ndim == 1):
            x.unsqueeze(0)
        inp = []
        inp.append(x[0, :-1])
        if(len(x) > 1):
            inp.append(x[-2, :-1])
        else:
            inp.append(x[0, :-1])
        inp = np.hstack(inp)
        X.append(inp)
    return np.vstack(X)

for x in preProcessed:
    y.append(x[-1,-1])
y = np.vstack((y))

X = generateInputs(preProcessed)

Xtest = generateInputs(preTrained)

patientsLabels = np.vstack(patientLabels)

#X = np.hstack((X[:,0][:,np.newaxis], np.sum(X[:, 1:31], axis = 1)[:,np.newaxis], X[:,31][:,np.newaxis], np.sum(X[:, 32:], axis = 1)[:,np.newaxis]))
'''
mseTrain = []
mseTest = []
for x in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=x)
    
    reg = linear_model.Ridge(alpha = 0)
    reg = reg.fit(X_train, y_train)
    #reg = GradientBoostingRegressor(random_state = x, n_estimators = 100, max_depth=3).fit(X_train,y_train.squeeze())
    
    y_pred = reg.predict(X_test)
    y_pred_train = reg.predict(X_train)
    mseTrain.append(np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)))
    mseTest.append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
#print(np.average(mseTrain))
#print(np.average(mseTest))
'''
reg = LinearRegression().fit(X, y)
labels = reg.predict(Xtest)
#defaultOut = np.average(finalPANS)

#naive model
labels = np.sum(Xtest[:, 32:], axis = 1)[:,np.newaxis]

#modifying base model with 
labels = labels.squeeze().astype(float)
oldlabels = labels.copy()
#print(labels)
for x in range(len(GoingtoModelInds)):
    #print(x)
    labels[GoingtoModelInds[x]] = predictions[x]
#print(labels - oldlabels)
labels = labels[:,np.newaxis]
#labels[ShortInds] = defaultOut

#blanket submitting the average of the less than 5 classes isn't going to work

outputs = np.hstack((patientsLabels, labels))
out = pd.DataFrame(outputs)

out.columns = ["PatientID", "PANSS_Total"]
out["PatientID"]= out["PatientID"].astype(int)
out.to_csv("regression.csv", index = False)




