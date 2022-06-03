import hd_clustering
import torch
import numpy as np
from sklearn import datasets
import json
import typing
import matplotlib.pyplot as plt
from copy import deepcopy

DATA_LOC = '../Conventional_Data/'


def get_mnist_dataset(max_samples):

    dl = hd_clustering.Dataloader(
        dir='mnist', dataset='mnist', data_loc=DATA_LOC)
    nFeatures, nClasses, traindata, trainlabels, testdata, testlabels = dl.getParam()
    traindata = traindata[:max_samples]
    trainlabels = trainlabels[:max_samples]

    return nFeatures, nClasses, traindata, trainlabels, testdata, testlabels


def get_isolet_dataset(max_samples):

    dl = hd_clustering.Dataloader(
        dir='isolet', dataset='isolet', data_loc=DATA_LOC)
    nFeatures, nClasses, traindata, trainlabels, testdata, testlabels = dl.getParam()
    traindata = traindata[:max_samples]
    trainlabels = trainlabels[:max_samples]

    return nFeatures, nClasses, traindata, trainlabels, testdata, testlabels


def get_iris_dataset(max_samples):

    iris = datasets.load_iris()
    X = iris.data[:max_samples]
    y = iris.target[:max_samples]

    nFeatures = X.shape[1]
    nClasses = 3

    return nFeatures, nClasses, X, y, X, y


def get_prob_tables_from_json(path) -> typing.Dict[str, typing.List]:

    with open(path, 'r', encoding='utf8') as f:
        prob_tables = json.load(f)

    return prob_tables


def get_prob_table_from_prob(prob: float, bits: int):

    prob_table = [[prob * 100 / 2, prob * 100 / 2] for _ in range(bits**2)]
    prob_table[0] = [prob * 100, 0.]
    prob_table[-1] = [0., prob * 100]

    return prob_table


def run_clustering(nFeatures: int,
                   nClasses: int,
                   traindata: np.ndarray,
                   trainlabels: np.ndarray,
                   testdata: np.ndarray,
                   testlabels: np.ndarray,
                   bits: int,
                   dim: int,
                   epochs: int,
                   prob_table: typing.List[typing.List[float]],
                   flip_inference_only: bool = False):

    clusters = nClasses
    features = nFeatures
    model = hd_clustering.QuantHD_cluster(clusters, features, bits, dim=dim)

    history = []

    max_acc = 0.
    for epoch in range(epochs):

        model.fit(torch.tensor(traindata.astype(np.float32)),
                  epochs=1, init_model=(not epoch), labels=trainlabels)

        if flip_inference_only:

            copied_model = deepcopy(model)
            copied_model.random_bit_flip_by_prob(prob_table)

            ypred = copied_model(torch.tensor(traindata.astype(np.float32)))
            train_acc = (ypred == torch.tensor(trainlabels)
                         ).sum().item() / len(ypred)
        else:

            model.random_bit_flip_by_prob(prob_table)

            ypred = model(torch.tensor(traindata.astype(np.float32)))
            train_acc = (ypred == torch.tensor(trainlabels)
                         ).sum().item() / len(ypred)

        #print(epoch, train_acc)
        history.append(train_acc)

        #max_acc = max(max_acc, train_acc)
        if max_acc < train_acc:

            max_acc = train_acc
            best_model = deepcopy(model)

    print(max_acc)

    return history


def plot_histories(histories):

    plt.figure(figsize=(10, 7))
    for history in histories:
        plt.plot(history)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
