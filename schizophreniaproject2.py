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
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

#for graphing
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



dataframes = ["Study_A.csv", "Study_B.csv", "Study_C.csv", "Study_D.csv"]

trains = []
for x in dataframes:
    trains.append( pd.read_csv(x))
train = pd.concat(trains)

#no duplicate patient IDs here
baseline = ps.sqldf("select * from train where VisitDay == 0 and LeadStatus = 'Passed'")

minse = ps.sqldf("select PatientID, min(AssessmentID) as vis from baseline group by PatientID")
aggedBaseline = ps.sqldf("select s.* from baseline s inner join minse on s.PatientID = minse.PatientID and s.AssessmentID = minse.vis")

basePy = aggedBaseline.to_numpy()

basePy = basePy[:, 8:-2]

rawPy = basePy


# we need to make sure our dataset is scaled properly!
basePy = StandardScaler().fit_transform(basePy)

x = []
y = []

for i in range(1, 11):
    kmeans_model = KMeans(n_clusters=i, init= "k-means++", random_state=1).fit(basePy)
    labels = kmeans_model.labels_
    #print(kmeans_model.inertia_)
    y.append(sum(np.min(cdist(basePy, kmeans_model.cluster_centers_,
                                        'euclidean'), axis=1)) / basePy.shape[0])
    x.append(i+1)
    

#looks like 4 clusters are the best. The cdist stops decreasing by such large 
#margins of over 2k after 4 clusters are wrought
clusts = 3

#plt.plot(x,y)
#plt.show()

kmeans_model = KMeans(n_clusters=clusts, init= "k-means++", random_state=1).fit(basePy)
labels = kmeans_model.labels_
#sns.heatmap(data =kmeans_model.cluster_centers_)

#as a sanity check, let's see if an agglomerative clustering approach produces
#roughly equivalent clusters


model = AgglomerativeClustering(distance_threshold = None, n_clusters = clusts, linkage = "ward")
model = model.fit(basePy)
x = model.labels_
means = []
covs = []
for i in range(clusts):
    inds = np.where(x == i)
    cluster = basePy[inds]
    means.append(np.mean(cluster, axis = 0))
    mat = np.absolute(np.cov(cluster.astype(float), rowvar= False))
    #np.fill_diagonal(mat, 0)
    covs.append(np.max(mat))
    

heat = means.pop(0)
for x in means:
    heat = np.vstack((heat, x))
print(heat.shape)
#sns.heatmap(data = heat.astype(float))

#as a final sanity check, let's do some PCA graphing 
pca = PCA(2)
proj = pca.fit_transform(basePy)
#'''
plt.scatter(rawPy[:,0], rawPy[:,1], c=labels,
            cmap="rainbow", alpha = 0.2)
plt.colorbar()
plt.show()
#'''




