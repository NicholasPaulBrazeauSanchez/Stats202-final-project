import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandasql as ps
from scipy.stats import multivariate_normal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from fitter import Fitter, get_common_distributions, get_distributions
from scipy import stats

#NOTE THAT YOU MUST PIP INSTALL PINGOUIN TO MAKE FULL USE OF THE CODE
import pingouin



dataframes = ["Study_A.csv", "Study_B.csv", "Study_C.csv", "Study_D.csv"]

#we can also feed our train dataframes different studies to see whether or not
#there's an in-study population effect
#dataframes = [ "Study_C.csv"]

trains = []
for x in dataframes:
    trains.append( pd.read_csv(x))
train = pd.concat(trains)

#no duplicate patient IDs here
control = ps.sqldf("select * from train where LeadStatus = 'Passed' and TxGroup == 'Control'")
treatment = ps.sqldf("select * from train where LeadStatus = 'Passed' and TxGroup == 'Treatment'")
controlPy = control.to_numpy()
treatmentPy = treatment.to_numpy()

controlPy = control.to_numpy()

#we acquire the first and final readings on a patient by patient basis across 
#our studies 
def preProcess(inp):
    basePy = []
    newPy = []
    patient = inp[0,2]
    minn = inp[0, 8:39]
    maxx = inp[0, 8:39]
    for x in inp:
        if(x[2] == patient):
            maxx = x[8:39]
        else:
            basePy.append(minn)
            newPy.append(maxx)
            patient = x[2]
            minn = x[8:39]
            maxx = x[8:39]
    basePy.append(minn)
    newPy.append(maxx)
    return np.vstack(basePy), np.vstack(newPy)

#this allows us to observe clusters of observation sums, P, N and G
def addsSums(inp):
    Pscores = np.sum(inp[:, :7], axis = 1)
    Nscores = np.sum(inp[:, 7:14], axis = 1)
    Gscores = np.sum(inp[:, 14:30], axis = 1)
    return np.hstack((inp, Pscores[:, np.newaxis], Nscores[:, np.newaxis], Gscores[:, np.newaxis]))

basePANStreat, newPANStreat = preProcess(treatmentPy)




basePANStreat = addsSums(basePANStreat.astype(float))

newPANStreat = addsSums(newPANStreat.astype(float))

basePANScontrol, newPANScontrol = preProcess(controlPy)

basePANScontrol = addsSums(basePANScontrol.astype(float))

newPANScontrol = addsSums(newPANScontrol.astype(float))

#plt.hist(basePANStreat, 50)

#plt.hist(newPANStreat, 50)

#plt.hist(newPANScontrol, 50)

statistics = []
pvals = []

for x in range(len(newPANScontrol[0])):
    #variable 26 is interestin
    treatPANSvar = newPANStreat[:,x]
    controlPANSvar = newPANScontrol[:,x]
    treatDiff = newPANStreat[:,x] - basePANStreat[:,x]
    controlDiff = newPANScontrol[:,x] - basePANScontrol[:,x]
    #'''
    statisticsDist = []
    statisticsDist.append(np.mean(treatPANSvar))
    statisticsDist.append(np.mean(controlPANSvar))
    statisticsDist.append(np.mean(treatDiff))
    statisticsDist.append(np.mean(controlDiff))
    statistics.append(statisticsDist)
   
    pvalues = []
    
    pvalues.append(stats.kstest(controlPANSvar, treatPANSvar)[1] < 0.05)
    pvalues.append(stats.ttest_ind(controlPANSvar, treatPANSvar, equal_var = False)[1] < 0.05)
    pvalues.append(stats.kstest(controlDiff, treatDiff)[1] < 0.05)
    pvalues.append(stats.ttest_ind(controlDiff, treatDiff, equal_var = False)[1] < 0.05)
    pvals.append(pvalues)

for x in range(len(pvals)):
    print(x)
    print( True in pvals[x])

#we found that the 25th datapoint seemed to show a pretty marked difference 
#in t statistics for the differences. Let's graph it out 
plt.hist(newPANScontrol[:,25] - basePANScontrol[:,25])

plt.hist(newPANStreat[:,25] - basePANStreat[:,25], alpha = 0.4)


#One last idea: let's use a multivariate ttest to compare the final symptom distributions
#at a macroscopic level!
#NOTE THAT YOU MUST PIP INSTALL PINGOUIN TO MAKE FULL USE OF THE CODE
print(pingouin.multivariate_ttest(newPANScontrol[:,:-4], newPANStreat[:,:-4]))
controlDiffs = newPANScontrol[:,:-4] - basePANScontrol[:,:-4]
treatDiffs = newPANStreat[:,:-4] - basePANStreat[:,:-4]
print(pingouin.multivariate_ttest(controlDiffs, treatDiffs))
    







