import math
from typing import Union
from scipy import stats
import numpy as np
import random

import torch

from . import Encoder

# X should be the class matrix of shape nClasses * D
# 0 is mapped to most lowest value and 2^(bits)-1 highest
def quantize(X, bits):
    Nbins = 2**bits
    # ultimate cheess
    bins = [(i / Nbins) for i in range(Nbins)]
    # notice the axis along which to normalize is always the last one
    nX = stats.norm.cdf(stats.zscore(X, axis=X.ndim-1))
    nX = np.digitize(nX, bins) - 1
    #print("Max and min bin value:", np.max(nX), np.min(nX))
    #print("Quantized from ", X)
    #print("To", nX)
    nX = torch.tensor(nX.astype(np.float32))
    return nX


# hyperdimensional clustering algorithm
class QuantHD_cluster(object):

    def __init__(self, clusters : int, features : int, bits : int, dim : int = 4000):

        self.clusters = clusters
        self.bits = bits
        self.dim = dim

        self.model = torch.zeros(self.clusters, self.dim)
        self.quantized_model = torch.zeros(self.clusters, self.dim)
        self.encoder = Encoder(features, dim=self.dim)

    def __call__(self, x : torch.Tensor, encoded : bool = False):

        return self.scores(x, encoded=encoded).argmax(1)

    def predict(self, x : torch.Tensor, encoded : bool = False):

        return self(x)

    def probabilities(self, x : torch.Tensor, encoded : bool = False):

        return self.scores(x, encoded=encoded).softmax(1)
    
    def dist(self, x1 : torch.Tensor, x2 : torch.Tensor):

        x1_norm = x1 / x1.norm(dim=1)[:, None]
        x2_norm = x2 / x2.norm(dim=1)[:, None]

        return torch.ones(x1.shape[0], x2.shape[0]) - torch.mm(x1_norm, x2_norm.transpose(0, 1))

        #return torch.cdist(x1, x2, 1) / x1.shape[1]

    def scores(self, x : torch.Tensor, encoded : bool = False):

        '''
        h = x if encoded else self.encode(x)
        #return 1 - torch.cdist(h.sign(), self.quantized_model.sign(), 0)/self.dim

        h = quantize(h, self.bits) 

        h_norm = h / h.norm(dim=1)[:, None]
        model_norm = self.quantized_model / self.quantized_model.norm(dim=1)[:, None]

        return torch.mm(h_norm, model_norm.transpose(0, 1))
        '''
        
        h = x if encoded else self.encode(x)
        #return 1 - torch.cdist(h.sign(), self.quantized_model.sign(), 0)/self.dim

        h = quantize(h, self.bits)

        return -self.dist(h, self.quantized_model)

    def encode(self, x : torch.Tensor):
        
        return self.encoder(x)
    
    def model_projection(self):

        if self.bits == -1:
            return -1

        for i in range(self.clusters):
            self.quantized_model[i] = quantize(self.model[i], self.bits)
        
        return -1

    def fit(self,
            x : torch.Tensor,
            encoded : bool = False,
            epochs : int = 40,
            batch_size : Union[int, float, None] = None,
            adaptive_update : bool = True,
            binary_update : bool = False,
            init_model : bool = True,
            labels=None):

        h = x if encoded else self.encode(x)
        n = h.size(0)

        # converts batch_size to int
        if batch_size is None:
            batch_size = n
        if isinstance(batch_size, float):
            batch_size = int(batch_size*n)

        # initializes clustering model
        if init_model:
            '''
            # Random select
            idxs = torch.randperm(n)[:self.clusters]
            self.model.copy_(h[idxs])
            '''
            '''
            # Random
            self.model.copy_(torch.empty(self.clusters, self.dim).uniform_(0.0, 2*math.pi))
            '''
            # k-means++
            self.model.copy_(torch.zeros(self.clusters, self.dim))
            new_center = random.randint(0, len(h) - 1)
            self.model[0] = h[new_center]
            if labels is not None:
                print(labels[new_center])
            dist = torch.zeros(len(h))
            for k in range(self.clusters - 1):
                dist_pair = self.dist(h, self.model)

                for i in range(len(h)):
                    
                    dist[i] = dist_pair[i, :k+1].min()
                
                prob_distribution = dist**2
                prob_distribution /= prob_distribution.sum()
                new_centers = np.random.choice([i for i in range(len(h))], 10,
                    p=prob_distribution.numpy())

                new_center = new_centers[0]
                
                self.model[k + 1] = h[new_center]
                if labels is not None:
                    print(labels[new_center])
            
            self.model_projection()

            print(self.model)
            print(self.quantized_model)

        # previous_preds will store the predictions for all data points
        # of the previous iteration. used for automatic early stopping
        previous_preds = torch.empty(n, dtype=torch.long,
                device=self.model.device).fill_(self.clusters)

        # starts iterative training
        for epoch in range(epochs):
            # found_new will stay False if, during current epoch, no data
            # point changed their cluster comparing it to the previous epoch
            found_new = False

            new_model = torch.zeros(self.clusters, self.dim)
            cnt = torch.zeros(self.clusters)
            for i in range(0, n, batch_size):
                h_batch = h[i:i+batch_size]
                scores = self.scores(h_batch, encoded=True)
                max_score, preds = scores.max(1)

                # if no new predictions durent current iteration (batch), skip
                if (previous_preds[i:i+batch_size] == preds).all():
                    continue

                # updates previous_preds vector
                found_new = True
                previous_preds[i:i+batch_size] = preds

                # updates each clustering model
                for lbl in range(self.clusters):
                    h_batch_lbl = h_batch[preds == lbl]
                    #self.model[lbl] += h_batch_lbl.sum(0)
                    new_model[lbl] += h_batch_lbl.sum(0)
                    cnt[lbl] += len(h_batch_lbl)


                '''
                # if using binary update, clustering update will used
                # binarized datapoints instead
                if binary_update:
                    h_batch = h_batch.sign()

                # if using adaptive update, the respective alpha scaler will
                # be taken into account
                if adaptive_update:
                    std, mean = torch.std_mean(scores, 1)
                    alpha = ((max_score - mean)/std).unsqueeze(1)
                    h_batch = alpha*(h_batch.sign())

                # updates each clustering model
                for lbl in range(self.clusters):
                    h_batch_lbl = h_batch[preds == lbl]
                    self.model[lbl] += h_batch_lbl.sum(0)
                '''
            #print(cnt)
            for lbl in range(self.clusters):
                self.model[lbl].copy_(new_model[lbl] / cnt[lbl])

            # early stopping when the model converges
            if not found_new:
                break
        
        self.model_projection()

        return self

    def to(self, *args):

        self.model = self.model.to(*args)
        self.quantized_model = self.quantized_model.to(*args)
        self.encoder = self.encoder.to(*args)
        return self
