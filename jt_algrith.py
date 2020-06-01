# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:33:06 2020

@author: Leon
"""
#haha
#hehe
import copy
import numpy as np
import math
from scipy.special import gamma
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#data processing: delete duplication itme
def dataDeleteDuplication(data_sample):
    data_arr = np.array(data_sample)
    data_sample = np.array(list(set([tuple(t) for t in data_arr])))
    return data_sample

def dataSorted(data_sample):
    data_sample = sorted(data_sample, key=(lambda x:x[0])) 
    return data_sample

#data processing: getmode
def getMode(data_point, thrhd):
    data_frequency = np.array(data_point[0])
    data_point[0] = data_frequency//thrhd*thrhd
    return data_point

#data lofar processing 
def dataProcessing(data_lofar, thrhd):
    for i in range(len(data_lofar)):
        data_lofar[i] = dataDeleteDuplication(data_lofar[i])
        data_lofar[i] = dataSorted(data_lofar[i])
        for j in range(len(data_lofar[i])):
            data_lofar[i][j] = getMode(data_lofar[i][j], thrhd)
    return data_lofar

#make-up lofar data with zero 
def zeroSupplement(data):
    frequency_set = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j][0] not in frequency_set:
                frequency_set.append(data[i][j][0])
    frequency_set = sorted(set(frequency_set))
    data_zero_lofar = []
    
    for k in range(len(frequency_set)):
        data_zero_lofar.append([frequency_set[k] ,0])
    
    for m in range(len(data)):
        temp = copy.deepcopy(data_zero_lofar)
        for n in range(len(data[m])):
            lofar_index = frequency_set.index(data[m][n][0])
            temp[lofar_index] = data[m][n]
        data[m] = temp
    return data

#extract value from lofar data(including frequency an value)
def extractValue(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j][1]
    return data

#dimensionality reduction    
def dimen_reduction(method, dimension, data):
    if method == 'PCA':
        data_reduc = dimen_reduc_PCA(dimension, data)
    elif method == 'TSNE':
        data_reduc = dimen_reduc_TSNE(dimension, data)
    return data_reduc

#k-means clustering algorithm
def kmeans(k, data):
    km = KMeans(n_clusters = k)  #set k clusters
    km.fit(data)  # k-means clustering
    labels = km.labels_  # get label
    return labels


#DBSCAN clustering algorithm
def dbscan(data):
    dataSet = dimen_reduction('PCA', 10, data)
    max_item = max(max(row) for row in dataSet)
    min_item = min(min(row) for row in dataSet)
    data_prod = np.prod(max_item-min_item)
    eps = math.pow(data_prod*5*gamma(0.5*len(dataSet[1]) + 1)/(len(dataSet)*math.sqrt(math.pow(math.pi, len(dataSet[1])))), 1/len(dataSet[1]))
    db = DBSCAN(eps = eps, min_samples = 5).fit(dataSet)
    labels = db.labels_
    return labels

#GaussianMixture clustering algorithm
def GMM(k, data):
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=28)
    gmm.fit(data)
    labels = gmm.predict(data)
    return labels

#BIRCH clustering algorithm
def brich(k, data):
    clf = Birch(n_clusters=k, threshold=0.1,
                branching_factor=10)  # n_clusters初始值不知道可以设置为None
    clf.fit(data)
    labels = clf.predict(data)
    return labels

#FCM clustering algorithm
def FCM(k, data):
    m = 2
    eps = 10
    membership_mat = np.random.random((len(data), k))
    membership_mat = np.divide(membership_mat, np.sum(membership_mat, axis=1)[:, np.newaxis])

    while True:
        working_membership_mat = membership_mat ** m
        Centroids = np.divide(np.dot(working_membership_mat.T, data), np.sum(working_membership_mat.T, axis=1)[:, np.newaxis])
        n_c_distance_mat = np.zeros((len(data), k))
        for i, x in enumerate(data):
            for j, c in enumerate(Centroids):
                n_c_distance_mat[i][j] = np.linalg.norm(x-c, 2)        
        new_membership_mat = np.zeros((len(data), k))        
        for i, x in enumerate(data):
            for j, c in enumerate(Centroids):
                new_membership_mat[i][j] = 1. / np.sum((n_c_distance_mat[i][j] / n_c_distance_mat[i]) ** (2 / (m - 1)))
        if np.sum(abs(new_membership_mat - membership_mat)) < eps:
            break
        membership_mat =  new_membership_mat
        labels = np.argmax(new_membership_mat, axis = 1)
    return labels

#StandardSca clustering algorithm
def SC(k, data):
    SC = SpectralClustering(n_clusters = k, n_neighbors = 10)
    SC.fit(data)
    labels = SC.fit_predict(data)
    return labels

#PCA dimensionality reduction
def dimen_reduc_PCA(n, dataSet):
    pca = PCA(n_components = n)
    dataSet = pca.fit_transform(dataSet)
    return dataSet

#TSNE dimensionality reduction
def dimen_reduc_TSNE(n,dataSet):
    tsne = TSNE(n_components = n)
    dataSet = tsne.fit_transform(dataSet)
    return dataSet

# return centers of clusters
def getDimAndGetCenter(data,lableList):
    data = dimen_reduction('PCA', 3, data)
    k =np.max(lableList)+1
    tmps = [[]for l in range(k)]
    for j in range(k):
        for i in range(len(lableList)):
            if lableList[i] == j:
                tmps[j].append((data[i]))
    np_tmps = np.array(tmps)
    centers =np_tmps.mean(1)
    return data,centers

def clustering_method_choose(method, data, k):
    if method == 'k-means':
        label_pred = kmeans(k, data)
    elif method == 'DBSCAN':
        label_pred = dbscan(data)
    elif method == 'GMM':
        label_pred = GMM(k, data)
    elif method == 'BRICH':
        label_pred = brich(k, data)
    elif method == 'FCM':
        label_pred = FCM(k, data)
    elif method == 'SC':
        label_pred = SC(k, data)
    return label_pred

def demension_reduction_method_choose(method, data, centroids):
    if method == 'PCA':
        draw = dimen_reduction('PCA', 3, data)
        draw_centroids = dimen_reduction('PCA', 3, centroids)
    elif method == 'TSNE':
        draw = dimen_reduction('TSNE', 3, data)
        draw_centroids = dimen_reduction('TSNE', 3, centroids)
    return draw, draw_centroids

def getJson(jsonData):
    res = {}
    try:
        dataAfterProcessing = dataProcessing((jsonData['data']), jsonData['thrhd'])
        dataAfterZeroSupp = zeroSupplement(dataAfterProcessing)
        dataValueExtract = extractValue(dataAfterZeroSupp)
        # choose clusteing method:k-means;DBSCAN;GMM;birch;FCM;SC
        label_pred = clustering_method_choose(jsonData['method'], dataValueExtract, jsonData['K'])
        data, draw_centroids = getDimAndGetCenter(dataValueExtract, label_pred)

        res['label'] = str(list(label_pred)).replace(" ","")
        res['centers'] = str(list(draw_centroids)).replace("  ",",").replace(" ","").replace(",,",",").replace("array(","").replace(")","")
        res['data'] = str(list(data)).replace("  ",",").replace(" ","").replace(",,",",").replace("array(","").replace(")","")

    except Exception as e:
        print(e)
        res['error'] = "Algrithm error :"+str(e)
        pass
    # print(res)
    return res
