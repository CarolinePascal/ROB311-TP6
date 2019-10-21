from sklearn import svm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
import random as rd
import time
import pickle


def load_data(PATH):
    with open(PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        images = []
        labels = []
        rowcounter = 0
        for row in csv_reader:
            if(rowcounter == 0):
                rowcounter+=1;
                continue
            images.append(np.array(row[0:len(row)-1]).astype(np.float))
            labels.append(row[-1])
    return(np.array(labels),np.array(images))

def init_centroids_random(k, data):
    centroid_indexes = np.floor(np.random.random_sample((k,))*len(data)).astype(int)
    
    return np.take(data,centroid_indexes, axis=0)

def init_centroids_each_class(k, data, labels):
    centroid_indexes= np.empty(k)
    for i in range(k):
        centroid_indexes[i] = np.where(labels == str(i))[0][0]

    centroid_indexes = centroid_indexes.astype(int)
    centroids = np.take(data,centroid_indexes, axis=0)

    return centroids

def compute_clusters(centroids, data):
    clusters = np.empty(len(data))
    for i, point in enumerate(data):
        diff = centroids-point
        distances = np.sum(diff*diff, axis=1)
        clusters[i] = np.argmin(distances)

    return clusters

def comp_new_centroids(k, data, clusters):
    centroids = np.empty((k, len(data[0])))
    for i in range(k):
        cluster_indexes = np.where(clusters == i)[0]
        cluster_data = np.take(data, cluster_indexes, axis=0)
        centroids[i] = np.mean(cluster_data, axis=0)

    return centroids


def main():
    k= 10
    labels_train, images_train = load_data('optdigits_train.csv')
    labels_test, images_test = load_data('optdigits_test.csv')

    centroids= init_centroids_random(10, images_train)
    #centroid_indexes = init_centroids_each_class(images_train, labels_train)
    old_centroids= np.zeros(centroids.shape)

    nb_it = 0
    nb_max_it = 100
   
    while not np.all(centroids == old_centroids) and nb_it < nb_max_it:
        clusters = compute_clusters(centroids, images_train)
        old_centroids = centroids
        centroids = comp_new_centroids(k, images_train, clusters)
        nb_it+=1

    print("k-mean algorithm was ran in "+str(nb_it)+"iterations")

    ## prediction
    clusters_test = np.empty(len(labels_test))
    for i, image in enumerate(images_test):
        diff = centroids-image
        distances = np.sum(diff*diff, axis=1)
        clusters_test[i] = np.argmin(distances)

    print(clusters_test)


main()
