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
import torch.optim as optim
from sklearn.preprocessing import normalize

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
#testData[:, 8:-1] = normalize(testData[:, 8:-1], axis = 0)

trainData = train.to_numpy()
#trainData[:, 8:-2] = normalize(trainData[:, 8:-2], axis = 0)

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
        valueRanges = [35,77,119,161, 203]
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
outTimes = [x[-1,0] for x in processed]
lengths = [len(x) for x in X]


    


#predictions 
#processed actually has the time series data we're interested in using
#173 datapoints using the training data

viewPretrained = [x for x in preTrained if len(x) < 4 and len(x) > 1]
ShortInds = []
LongInds = []
GoingtoModelInds = []
modelData = []
finalPANS = []
outTimesTest = []
lengthsTest = []
for x in range(len(preTrained)):
    if len(preTrained[x]) < 5:
        ShortInds.append(x)
        finalPANS.append(preTrained[x][-1,-1])
    else: 
        LongInds.append(x)
        if(preTrained[x][-1,0] >= 150 and preTrained[x][-2,0] >= 110):
            GoingtoModelInds.append(x)
            modelData.append(preTrained[x])
            lengthsTest.append(len(preTrained[x]))
            outTimesTest.append(preTrained[x][-1,0]+42)
            

#naive model
Labels = []
for x in preTrained:
    Labels.append(x[-1, -1])
Labels = np.asarray(Labels).astype(float)


#we're trying neural networks now



predictions = []

def collate_fn(examples):


    #    Group by tensor type
    sequences, lens, outputTimes, labels = zip(*examples)

    # Merge into batch tensors
    sequences = pad_sequence(sequences)
    
    return (sequences, lens, outputTimes, labels)

class SchizophreniaDataset(Dataset):
    def __init__(self, labels, outputTimes, sequences, lens):
        self.labels = labels
        self.outputTimes = outputTimes
        self.sequences = sequences
        self.lens = lens
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        tense = torch.from_numpy(self.sequences[index].astype(int))
        
        if(len(tense.size()) == 1):
            tense = torch.unsqueeze(tense, 0)

        return (tense, self.lens[index], self.outputTimes[index], self.labels[index])
    
class GruClass(torch.nn.Module):
    def __init__(self, hidden_size, drop_prob= 0.0):
        super(GruClass, self).__init__()
        self.hidden_size = hidden_size
        self.GRU = torch.nn.LSTM(32, hidden_size, 2, dropout = drop_prob)
        self.output1 = torch.nn.Linear(hidden_size+1, 1)
        self.output2 = torch.nn.Linear(5, 1)
        self.Relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(drop_prob)
    
    def forward(self, seqs, times, lens):
        inps = torch.nn.utils.rnn.pack_padded_sequence(seqs, lens, enforce_sorted = False)
        _,(out,_) = self.GRU(inps)
        out = torch.cat((out[-1], times.unsqueeze(1)), axis = 1)
        output = (self.output1(out))
        return output
        
dataset = SchizophreniaDataset(y, outTimes, X, lengths)
testDataset = SchizophreniaDataset(lengthsTest, outTimesTest, modelData, lengthsTest)
train_set, val_set = torch.utils.data.random_split(dataset, [int(0.8*dataset.__len__()), dataset.__len__()-int(0.8*dataset.__len__())])

#LRs = [0.0002,0.0003,0.0004,0.0005]
LRs = [0.0004]

#dropouts = [0.05, 0.1, 0.15, 0.2]
dropouts = [0.05]


#hiddens = [3, 5, 7, 9]
hiddens = [12]
#12 works awesome

#batchSize = [4, 6, 8]
batchSize = [3]

bestLR = 0
bestDropouts = 0
bestHiddens = 0
bestbatchSize = 0
bestValLoss = 100000000
bestTrainLoss = 0

#for cross validation and hypeparameter searching
#'''
for lr in LRs:
    for drops in dropouts:
        for hids in hiddens:
            for batch in batchSize:
                bestValleLoss = 10000000
                batch_size = batch
                max_epochs = 175
                train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn = collate_fn)
                val_loader = torch.utils.data.DataLoader(val_set,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn = collate_fn)
                
                gru = GruClass(hids, drops)
                
                #7, 0.1 with 0.0003 lr and 4 batch size works best
                #
                
                
                optimizer = optim.Adam(gru.parameters(), lr=lr)
                
                losse = torch.nn.MSELoss()
                
                epoch = 0
                counter = 0
                while epoch != max_epochs:
                    
                    epoch += 1
                    torch.enable_grad() 
                    lossAvg = []
                    labelLens = []
                    for seqs, lens, times, labels in train_loader:
                            # Setup for forward
                            optimizer.zero_grad()
                            lens  = torch.Tensor(list(lens))
                            labels = torch.Tensor(list(labels))
                            times = torch.Tensor(list(times))
                            outputs = gru(seqs.float(), times, lens)
                            loss = losse(outputs.squeeze(1), labels.type(torch.FloatTensor))
                            loss_val = loss.item()*len(labels)
                            
                            loss.backward()
                            optimizer.step()
                            lossAvg.append(loss_val)
                            labelLens.append(len(labels))
                    trainLoss = sum(lossAvg)/sum(labelLens)
                    print(sum(lossAvg)/sum(labelLens))
                    torch.no_grad()
                    netLoss = 0
                    totalLoss = 0
                    
                    lossAvg = []
                    labelLens = []
                    for seqs, lens, times, labels in val_loader:
                        lens  = torch.Tensor(list(lens))
                        labels = torch.Tensor(list(labels))
                        times = torch.Tensor(list(times))
                        outputs = gru(seqs.float(), times, lens)
                        #print(outputs)
                        loss = losse(outputs.squeeze(1), labels.type(torch.FloatTensor)).item()
                        lossAvg.append(loss*len(labels))
                        labelLens.append(len(labels))
                       # print(labels)
                    thisLoss = sum(lossAvg)/sum(labelLens)
                    print(thisLoss)
                    if(thisLoss < bestValleLoss):
                        bestValleLoss = thisLoss
                    if(thisLoss - trainLoss > 10):
                        counter += 1
                        if(counter >= 200):
                            break
                    else:
                        counter = 0
                    print(epoch)
                    print(counter)
                if bestValleLoss < bestValLoss:
                    bestValLoss = bestValleLoss
                    bestLR = lr
                    bestDropouts = drops
                    bestHiddens = hids
                    bestbatchSize = batch
print(bestValLoss)
print(bestLR)
print(bestDropouts)
print(bestHiddens)
print(bestbatchSize) 
#'''

batch_size = 3
max_epochs = 175
train_loader = torch.utils.data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn = collate_fn)

test_loader = torch.utils.data.DataLoader(testDataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   collate_fn = collate_fn)

gru = GruClass(5, 0.05)


optimizer = optim.Adam(gru.parameters(), lr=LRs[0])

losse = torch.nn.MSELoss()

epoch = 0
counter = 0
while epoch != max_epochs:
    
    epoch += 1
    torch.enable_grad() 
    lossAvg = []
    labelLens = []
    for seqs, lens, times, labels in train_loader:
            # Setup for forward
            optimizer.zero_grad()
            lens  = torch.Tensor(list(lens))
            labels = torch.Tensor(list(labels))
            times = torch.Tensor(list(times))
            outputs = gru(seqs.float(), times, lens)
            loss = losse(outputs.squeeze(1), labels.type(torch.FloatTensor))
            loss_val = loss.item()*len(labels)
            
            loss.backward()
            optimizer.step()
            lossAvg.append(loss_val)
            labelLens.append(len(labels))
    trainLoss = sum(lossAvg)/sum(labelLens)
    print(sum(lossAvg)/sum(labelLens))
    torch.no_grad()
    netLoss = 0
    totalLoss = 0

def predict(data_loader, model):

    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for seqs,lens,times,labels in data_loader:
            lens  = torch.Tensor(list(lens))
            labels = torch.Tensor(list(labels))
            times = torch.Tensor(list(times))
            y_star = model(seqs.float(), times, lens)
            output = torch.cat((output, y_star), 0)

    return output

Predictions = predict(test_loader, gru).detach().numpy().squeeze()

            


y = []
X = []
preProcessed = [x for x in preProcessed if (x.ndim != 0 and x.shape[0] >= 2) ]

#X = np.hstack((X[:,0][:,np.newaxis], np.sum(X[:, 1:31], axis = 1)[:,np.newaxis], X[:,31][:,np.newaxis], np.sum(X[:, 32:], axis = 1)[:,np.newaxis]))

#defaultOut = np.average(finalPANS)

#modifying base model with 
Labels = Labels.astype(float)
oldlabels = Labels.copy()
#print(labels)
for x in range(len(GoingtoModelInds)):
    Labels[GoingtoModelInds[x]] = Predictions[x]
print(labels - oldlabels)
Labels = Labels[:,np.newaxis]
#labels[ShortInds] = defaultOut

#blanket submitting the average of the less than 5 classes isn't going to work
patientLabels = np.asarray(patientLabels)[:,np.newaxis]
#print(patientLabels[:,np.newaxis])
#print(Labels)
outputs = np.hstack((patientLabels, Labels))
out = pd.DataFrame(outputs)

out.columns = ["PatientID", "PANSS_Total"]
out["PatientID"]= out["PatientID"].astype(int)
out.to_csv("regression.csv", index = False)




