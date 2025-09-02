import os
import numpy as np
import pandas as pd
import statistics
import loader
import pickle
import gc
import collections
import utils
import correlationFunctions
import evaluation as evaluate

from pathlib import Path
from math import ceil
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances, pairwise
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.linear_model import SGDClassifier
from scipy import sparse
from gcn import GCNClassifier

from corel5k_evaluation import corel5k_class
from cifar10_evaluation import cifar10_class
from cars_evaluation import cars_class
from mnist_from_images_evaluation import mnist_from_images_class
from cub_200_evaluation import cub200_class

import umap
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tabulate import tabulate


def run(features, labels, folds, rks, graph_type, use_distance_matrix=False, classifier="opf"):
    results = []
    count = 0
    for test_index, train_index in folds:
        # print("train_index", len(train_index))
        # print("test_index", len(test_index))
        # print("Running for Fold", count)
        count += 1

        train_features = np.array([features[i] for i in train_index], dtype=np.float32)
        train_labels = [labels[i] for i in train_index]

        test_features = np.array([features[i] for i in test_index], dtype=np.float32)
        test_labels = [labels[i] for i in test_index]

        # X = pairwise_distances(train_features)
        # x = pairwise_distances(test_features)

        if classifier == "svm":
            # ICPR
            clf = svm.SVC(kernel="poly", degree=2, gamma=0.001, C=10)

            # clf = svm.SVC(kernel = 'linear', C = 0.1)

            # {'C': 100, 'gamma': 0.001, 'kernel': 'sigmoid'}
            # clf = svm.SVC(kernel="sigmoid", gamma=0.001, C=100)

            # {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
            # clf = svm.SVC(C=10, gamma=0.01, kernel='rbf')

            # clf = svm.SVC(
            #    C=10,gamma='scale',kernel='rbf'
            # )

            # clf = svm.SVC()
            clf.fit(train_features, train_labels)

            pred = clf.predict(test_features)
        elif classifier == "opf":
            # Init OPF
            opf = OPFClassifier(precomputed=use_distance_matrix)

            # Training
            opf.fit(train_features, train_labels)

            # Predict
            pred = opf.predict(test_features)
        elif classifier == "knn":
            neigh = KNeighborsClassifier(n_neighbors=20)

            neigh.fit(train_features, train_labels)

            pred = neigh.predict(test_features)
        elif classifier == "svm_test":
            log_reg = SGDClassifier(loss="squared_hinge", n_jobs=-1, alpha=1e-5)

            log_reg.fit(train_features, train_labels)

            pred = log_reg.predict(test_features)
        elif "gcn" in classifier:
            clf = GCNClassifier(classifier, rks, evaluate.dataset_size[dataset], number_neighbors=40, graph_type=graph_type)
            clf.fit(test_index, train_index, features, labels)
            pred = clf.predict()
        else:
            print("Classifier not found...")
            exit(1)

        # Append current result
        results.append([pred, test_labels, evaluation(pred, test_labels)])

    return results


def evaluation(pred, labels):
    acc = 0
    n = len(pred)
    for i in range(n):
        if pred[i] == labels[i]:
            acc += 1
    return acc / n


### MAIN ###

n_executions = 5
n_folds = 10

# dataset = "flowers"
dataset = "corel5k"

descriptors = {
    "flowers": ["CNN-ResNet", "CNN-DPNet", "CNN-SENet", "T2T-VIT24", "VIT-B16"],
    "corel5k": ["CNN-ResNet", "CNN-DPNet", "CNN-SENet", "T2T-VIT24", "VIT-B16"],
    "cub200": ["CNN-ResNet", "CNN-DPNet", "CNN-SENet", "T2T-VIT24", "VIT-B16"],
}

corMeasures = ["intersection", "jaccard", "jaccard_k", "kendalltau", "rbo"]

labels_size = {
    "mpeg7": 1400,
    "flowers": 1360,
    "corel5k": 5000,
}

# top_k = class_size
top_k = {
    "flowers": 80,
    "corel5k": 100,
    "cub200": 58,  # 11788imgs, 200 classes -> +- 58.84 imgs per class
}

classifiers = ["gcn_net", "gcn_sgc", "gcn_gat", "gcn_appnpnet", "gcn_arma"]

class_size = {"mpeg7": 20, "flowers": 80, "corel5k": 100}

L = {
    "flowers": 100,
    "corel5k": 100,
    "cub200": 100,
}

# Create Folder to store optimal thresholds dump
dir_path = os.path.join("optimal_thresholds")
try:
    os.makedirs(dir_path)
except FileExistsError:
    # directory already exists
    pass

# Create Results Folder
results_path = "."
#results_path = os.path.join("results", "experiments-cub200")
try:
    os.makedirs(results_path)
except FileExistsError:
    # directory already exists
    pass

# methods to consider
rerankings = ["", "_LHRR", "_RDPAC", "_BFSTREE"]

# Create matrix to store all results

matrix_lenght = 1 + (len(corMeasures) * 2)

# RUN MAIN LOOP
file = open(os.path.join(results_path, "100epochs_results_" + str(dataset) + ".txt"), "w+")
table = []
for c in classifiers:
    for graph_type in ["knn", "rec"]:
        for use_reranking in rerankings:
            if use_reranking != "":
                line_table = [c.replace("_", "-") + " + " + graph_type + " + " + use_reranking.replace("_", "")]
            else:
                line_table = [c.replace("_", "-") + " + " + graph_type]
            results_matrix = [[] for i in range(matrix_lenght)]

            # print("\n\nClassifier = {}".format(c), file=file)
            # *****
            # Descriptors
            for desc in descriptors[dataset][:]:
                if table == []:
                    table.append([""] + [desc for desc in descriptors[dataset]])
                # gcns require to load rks for computing the knn graph
                d = "datasets"
                descriptor_path = (
                    os.path.join(os.path.join(d, dataset), desc) + use_reranking + ".txt"
                )
                rks_original = loader.read_ranked_lists_file(descriptor_path, L[dataset])


                use_distance_matrix = False
                if ("VIT-B16" in desc) or ("T2T" in desc):
                    feature_path = (
                        os.path.join(
                            os.path.join("datasets", dataset + "-feat"), desc
                        )
                        + ".npy"
                    )
                    features = np.load(feature_path)
                else:
                    feature_path = (
                        os.path.join(
                            os.path.join("datasets", dataset + "-feat"), desc
                        )
                        + ".npz"
                    )
                    features = np.load(feature_path)["features"]

                if dataset == "mpeg7" or dataset == "flowers":
                    labels = [
                        i // class_size[dataset] for i in range(labels_size[dataset])
                    ]
                elif dataset == "corel5k":
                    labels = corel5k_class
                elif dataset == "cars":  # cars
                    labels = cars_class
                elif dataset == "mnist_from_images":
                    labels = mnist_from_images_class
                elif dataset == "cub200":
                    labels = cub200_class

                features = np.array(features, dtype=np.float32)

                # Split data in folds
                folds = utils.fold_split(features, labels, n_folds=n_folds)

                acc_without_ws = 0

                acc_list_without_ws = []
                for i in range(n_executions):
                    pred = run(
                        features,
                        labels,
                        folds,
                        rks_original,
                        graph_type,
                        use_distance_matrix=use_distance_matrix,
                        classifier=c,
                    )
                    acc_acum = 0
                    for p, l, acc in pred:
                        acc_acum += acc

                    acc_list_without_ws.append(acc_acum / n_folds * 100)

                acc_without_ws = str(round(statistics.mean(acc_list_without_ws),2)) + " +- " + str(round(np.std(acc_list_without_ws),4))

                #print(
                #    "\t\t{}\t\t{}%".format(desc, acc_without_ws), file=file,
                #)
                line_table.append(acc_without_ws)

                results_matrix[0].append(
                    round((statistics.mean(acc_list_without_ws)), 2)
                )

            gc.collect()
            table.append(line_table)

print(tabulate(table, tablefmt="latex"))
print(tabulate(table, tablefmt="latex"), file=file)
file.close()
