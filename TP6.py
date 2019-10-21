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
        for row in csv_reader:
            images.append(np.array(row[0:len(row)-1]).astype(np.float))
            labels.append(row[-1])
    return(np.array(labels),np.array(images))

def init_centroids_random(k, data):
    """ @brief initialize centroids by taking k random points in the dataset 
    @param k : number of centroids to generate
    @param data: np array(k,m), images forming the dataset
    
    @return np array (k, n) : array of k centroids"""

    centroid_indexes = np.floor(np.random.random_sample((k,))*len(data)).astype(int)
    return np.take(data,centroid_indexes, axis=0)

def init_centroids_each_class(k, data, labels):
    """ @brief initialize centroids by taking the first point of each class in the dataset 

    @param k : number of centroids to generate
    @param data: np array(n,m), images forming the dataset
    @param labels: np array(n), the labels corresponding to the dataset
    
    @return np array (k, m) of k centroids"""

    centroid_indexes= np.empty(k)
    for i in range(k):
        centroid_indexes[i] = np.where(labels == str(i))[0][0]

    centroid_indexes = centroid_indexes.astype(int)
    centroids = np.take(data,centroid_indexes, axis=0)

    return centroids

def compute_fig(row):
    l = int(np.sqrt(len(row)))
    picture = np.empty((l,l))
    for i in range(l):
        picture[i,:] = row[i*l:(i+1)*l]

    return(picture)

def compute_clusters(centroids, data):
    """ @brief agregate all the points in the dataset into clusters

    @param centroids: np array (k,m) of k points acting as cluster centroids
    @param data: np array(n,m), images forming the dataset
    
    @return np array (n) of assigned cluster for each data point"""

    clusters = np.empty(len(data))
    for i, image in enumerate(data):
        diff = centroids-image
        distances = np.sum(diff*diff, axis=1)
        clusters[i] = np.argmin(distances)

    return clusters

def comp_new_centroids(k, data, clusters):
    """ @brief compute new centroids by averaging all the images in a cluster

    @param k: number of centroids to generate
    @param data: np array (n) of assigned cluster for each data point
    
    @return np array (k, m) of k new centroids"""

    centroids = np.empty((k, len(data[0])))
    for i in range(k):
        cluster_indexes = np.where(clusters == i)[0]
        cluster_data = np.take(data, cluster_indexes, axis=0)
        centroids[i] = np.mean(cluster_data, axis=0)

    return centroids

def results(k,data,labels,clusters):
    counters = np.zeros((k,k))
    for i,index in enumerate(clusters):
        counters[int(labels[i]),int(index)]+=1
    
    for i in range(k):
        argmax_c = np.argmax(counters[:,i])
        max_c = np.max(counters[:,i])
        sum_c = np.sum(counters[:,i])

        print("Predicted class "+str(i)+" : ")
        print("most common element : "+str(argmax_c)+ " (" + str(max_c) + " of " + str(sum_c)+")")
    
    return(counters)

def plot_confusion_matrix(numbers, cm, title='Confusion matrix', cmap=plt.cm.RdPu):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(numbers))
    plt.xticks(tick_marks, numbers)
    plt.yticks(tick_marks, numbers)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_mean_pictures(numbers, clusters,data):
    fig = plt.figure(figsize=(10,8))
    A = []
    for i in range(1,len(numbers)):
        A.append(fig.add_subplot(520+i))
    A.append(fig.add_subplot(5,2,10))

    for i,a in enumerate(A):
        a.imshow(compute_mean_picture(len(numbers),i,clusters,data),cmap='gray')
    plt.show()



def compute_mean_picture(k,index_cluster,clusters,data):
    pic_size = int(np.sqrt(len(data[0])))
    M = np.zeros((pic_size,pic_size))
    c=0

    for index in clusters:
        if(index==index_cluster):
            c+=1

    for i,index in enumerate(clusters):
        if(index==index_cluster):
            M += compute_fig(data[i])/c
        
    return(M)

def main():
    k= 10
    random_init = True
    labels_train, images_train = load_data('optdigits_train.csv')
    labels_test, images_test = load_data('optdigits_test.csv')

    if random_init:
        centroids= init_centroids_random(k, images_train)
    else:
        centroids = init_centroids_each_class(k, images_train, labels_train)

    old_centroids= np.zeros(centroids.shape)

    nb_it = 0
    nb_max_it = 100
   
    while not np.all(centroids == old_centroids) and nb_it < nb_max_it:
        clusters = compute_clusters(centroids, images_train)
        old_centroids = centroids
        centroids = comp_new_centroids(k, images_train, clusters)
        nb_it+=1

    print("k-mean algorithm was ran in "+str(nb_it)+" iterations")

    if random_init:
        #if initialization was done randomly, we try to reorganize the clusters in order to have all zeros in cluster zero
        # all ones in cluster one etc...

        # first we find out what if the most represented number in each cluster
        predicted_numbers = np.zeros(k)
        max_counts = np.zeros(k)
        for i in range(k):
            cluster = np.where(clusters == i)[0]
            true_labels = labels_train[cluster]
            unique, counts = np.unique(true_labels, return_counts=True)
            max_counts[i] = np.amax(counts)
            predicted_numbers[i] = unique[np.argmax(counts)]

        unique, counts = np.unique(predicted_numbers, return_counts = True)

        # if a number is the most represented in several clusters, 
        # then we keep the prediction for the best one and assign an other number to the remaining cluster

        while len(unique) != k:
            # finding out which numbers haven't been assigned to any cluster
            missing_elements = np.empty(0)
            for i in range(k):
                if not i in unique:
                    missing_elements = np.append(missing_elements, i)

            # running through ambiguous clusters, and assigning the worst cluster to one of the missing elements
            for ambiguous_class in unique[np.where(counts != 1)[0]]:
                ambiguous_class_indexes = np.where(predicted_numbers == ambiguous_class)[0]
                max_counts_ambiguous_class = np.take(max_counts, ambiguous_class_indexes)

                for i, candidate_class_index in enumerate(ambiguous_class_indexes):
                    if i != np.argmax(max_counts_ambiguous_class):
                        predicted_numbers[ambiguous_class_indexes[i]] = np.random.choice(missing_elements)
            
            unique, counts = np.unique(predicted_numbers, return_counts = True)


        # mapping the old cluster indexes to the new one
        
        new_clusters = clusters
        for i in range(k):
            i_indexes = np.where(clusters == i)
            new_clusters[i_indexes] = predicted_numbers[i]
        clusters = new_clusters


    ## prediction
    clusters_test = compute_clusters(centroids, images_test)

    numbers = ['0','1','2','3','4','5','6','7','8','9']

    matrix = results(k,images_test,labels_test,clusters_test)
    normalized_martix = matrix/matrix.sum(axis=1)

    plot_confusion_matrix(numbers,matrix)

    plot_confusion_matrix(numbers,normalized_martix, title='Normalized confusion matrix')

    plot_mean_pictures(numbers,clusters_test,images_test)

main()

