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

def init_centroids_random(data):
    centroids = np.floor(np.random.random_sample((10,))*len(data))
    return centroids

def init_centroids_each_class(data, labels):
    centroids = np.empty(10)
    for i in range(10):
        centroids[i] = np.where(labels == str(i))[0][0]

    return centroids

def compute_clusters(centroids_index, data):
    centroid_points = np.take(data,centroids_index, axis=0)
    clusters = np.empty(len(data))
    for i, point in enumerate(data):
        print(i)
        diff = centroid_points-point
        distances = np.sum(diff*diff)
        clusters[i] = np.argmin(distances)

    print(clusters)



labels_train, images_train = load_data('optdigits_train.csv')
init_centroids_random(images_train)
centroids_index = init_centroids_each_class(images_train, labels_train)
compute_clusters(centroids_index, images_train)