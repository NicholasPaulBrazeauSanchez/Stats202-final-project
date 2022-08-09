import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss



import pandasql as ps

    

#dataframes = ["Study_A.csv", "Study_B.csv", "Study_C.csv", "Study_D.csv"]
dataframes = ["Study_B.csv", "Study_A.csv"]
testData = ["Study_E.csv"]

test = pd.read_csv(testData[0])


trains = []
for x in dataframes:
    trains.append( pd.read_csv(x))
train = pd.concat(trains)


basePy = train.to_numpy()
passedInds = np.where(basePy[:,-1] == "Passed")

testPy = test.to_numpy()

#individual scores
nuPy = basePy[:, 7:-1]
testIDs = testPy[:, 5]
nuTestPy = testPy[:, 7:]




#aggregate
#nuPy = basePy[:, -2]
toLabel = {"Passed": 0, "Assign to CS": 1, "Flagged": 1}

def featureExtractor(inpTens):
    final = inpTens[1]
    final = np.hstack((final, inpTens[2:-1]))
    #final.append(np.sum(inpTens[2:-1]))
    return final
'''

def featureExtractor(inpTens):
    #print(inpTens)
    final = []
    final.append(inpTens[1])
    final.append(np.sum(inpTens[2:9]))
    final.append(np.sum(inpTens[9:16]))
    final.append(np.sum(inpTens[16:31]))
    return np.array(final)
'''
nuPy = np.hstack((basePy[:,2][:, np.newaxis], nuPy, basePy[:, -1][:, np.newaxis]))
testPy = np.hstack((testPy[:,2][:, np.newaxis], nuTestPy, testPy[:, -1][:, np.newaxis]))

def prePreProcess(inp, getLabs = True):
    patient = inp[0][0]
    labels = []
    sequences = []
    lens = []
    leng = 1
    runningAggregation = featureExtractor(inp[0])


    for x in inp[1:]:
        if getLabs:
            labels.append(toLabel[x[-1]])
        if(x[0] == patient):
            leng += 1
            #print(featureExtractor(x).shape)
            runningAggregation = np.vstack((runningAggregation, featureExtractor(x)))
        else:
            sequences.append(runningAggregation)
            #labels.append(toLabel[patientLabel])
            lens.append(leng)
            patient = x[0]
            patientLabel = x[-1]
            leng = 1
            runningAggregation = featureExtractor(x)
    sequences.append(runningAggregation)
    if getLabs:
        labels.append(toLabel[patientLabel])
    lens.append(leng)
    return sequences, labels

nuSeqs, nuLabels = prePreProcess(nuPy)
testSeqs, _ = prePreProcess(testPy, False)

print(nuSeqs[0].shape)
print(testSeqs[0].shape)


windowSize = 1

#we take a running average of 
def PreProcess(inp):
    X = []
    for x in inp:
        if(x.ndim == 1):
            x = x[np.newaxis, :]
        last = x[-1]
        first = x[0]
        for y in range(len(x)):
            seq = []
            windInd = y-windowSize
            while(windInd <= y+windowSize):
                if(windInd < 0):
                    seq.append(first)
                elif(windInd > len(x) - 1):
                    seq.append(last)
                else:
                    #print(x[windInd])
                    seq.append(x[windInd])
                #print(windInd)
                windInd += 1
                
            
            X.append(np.hstack(seq))
        #print(X)
    return X
        

X = PreProcess(nuSeqs)   
Xtest = PreProcess(testSeqs)  


    
y = np.hstack((nuLabels))
X = np.vstack((X))
Xtest = np.vstack((Xtest))
'''
X[:, 1:31] = X[:,32:62] - X[:, 1:31]
X[:, 63:] = X[:,32:62] - X[:, 63:]


training on sums of N, P, and G features
newCols = []
newCols.append(X[:, 0][:,np.newaxis])
newCols.append(np.sum(X[:, 1:8], axis = 1)[:,np.newaxis])
newCols.append(np.sum(X[:,8:16], axis = 1)[:,np.newaxis])
newCols.append(np.sum(X[:,16:31], axis = 1)[:,np.newaxis])
newCols.append(X[:,31][:,np.newaxis])
newCols.append(np.sum(X[:, 32:39], axis = 1)[:,np.newaxis])
newCols.append(np.sum(X[:,39:47], axis = 1)[:,np.newaxis])
newCols.append(np.sum(X[:,47:62], axis = 1)[:,np.newaxis])
newCols.append(X[:,62][:,np.newaxis])
newCols.append(np.sum(X[:,63:70], axis = 1)[:,np.newaxis])
newCols.append(np.sum(X[:,70:77], axis = 1)[:,np.newaxis])
newCols.append(np.sum(X[:,77:], axis = 1)[:,np.newaxis])
X = np.hstack((newCols))
'''
'''
newCols = []
newCols.append((np.sum(X[:, 1:31], axis = 1) - np.sum(X[:,32:62], axis = 1))[:,np.newaxis])
newCols.append(X[:,31:62])
newCols.append(np.sum(X[:, 32:62], axis = 1)[:,np.newaxis])
newCols.append((np.sum(X[:,32:62], axis = 1) - np.sum(X[:,63:], axis = 1))[:,np.newaxis])
X = np.hstack((newCols))

newCols = []
newCols.append((np.sum(Xtest[:, 1:31], axis = 1) - np.sum(Xtest[:,32:62], axis = 1))[:,np.newaxis])
newCols.append(Xtest[:,31:62])
newCols.append(np.sum(Xtest[:, 32:62], axis = 1)[:,np.newaxis])
newCols.append((np.sum(Xtest[:,32:62], axis = 1) - np.sum(Xtest[:,63:], axis = 1))[:,np.newaxis])

Xtest = np.hstack((newCols))
'''
Xflag = X[np.where(y == 1)]   
Xnorm = X[np.where(y == 0)]

train = []
test = []

#best: MLPClassifier 7,4
#best: reg = GradientBoostingClassifier(n_estimators = 500, max_depth=3, random_state=i).fit(X_train,y_train)
#1000, 3

learningRates = [ 0.1]
n_estimators = [200, 500, 1000]
max_depths = [2,3,4,5,6,7]
min_samples_splits = [0.1]
min_samples_leafs = [0.1]
max_features = [0.5, 0.7, 0.9]
bestLR = 0
bestestim = 0
bestdepth = 0
bestSamps = 0
bestleafs = 0
bestfeats = 0
bestVal = 1000000000

for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
        #X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size = 0.10, random_state = i)
        
        #reg = RandomForestClassifier(n_estimators = 150, max_depth=4, random_state=i).fit(X_train,y_train)

        #80, 4
        reg = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 200, max_depth = 4).fit(X_train,y_train)
        #reg = LinearDiscriminantAnalysis().fit(X_train, y_train)
        #reg = MLPClassifier(hidden_layer_sizes=(30, 2), max_iter=1000).fit(X_train,y_train)
        #reg = LogisticRegression().fit(X_train, y_train)
        
        y_train_preds = reg.predict_proba(X_train)
        y_test_preds = reg.predict_proba(X_test)
        #train.append(f1_score(y_train, y_train_preds, average='macro'))
        #test.append(f1_score(y_test, y_test_preds, average='macro'))
        #y_train_preds = np.full_like(y_train.astype(float), 0.34)
        #y_test_preds = np.full_like(y_test.astype(float), 0.34)
        train.append(log_loss(y_train, y_train_preds))
        test.append(log_loss(y_test, y_test_preds))
print(np.average(train))
print(np.average(test))


#(about .603 cross entropy)

reg = GradientBoostingClassifier(n_estimators = 200, max_depth=4, random_state=42).fit(X,y)
preds = reg.predict_proba(Xtest)[:,1]
#preds = np.full_like(preds, 0.2632)
print(preds.shape)
outputs = np.hstack((testIDs[:, np.newaxis], preds[:, np.newaxis]))
out = pd.DataFrame(outputs)
out.columns = ["AssessmentID", "LeadStatus"]
out.to_csv("classification.csv", index = False)



        
        



