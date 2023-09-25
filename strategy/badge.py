# original implementation is from https://github.com/JordanAsh/badge/blob/master/query_strategies/strategy.py
import gc

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch
from copy import deepcopy
from sklearn.metrics import pairwise_distances
import pdb
from scipy import stats
import math


class BADGE:
    def __init__(self, unlabeled_sample_ids, device="cpu"):

        self.unlabeled_sample_ids = unlabeled_sample_ids
        self.device = device

    # kmeans ++ initialization
    def init_centers_torch(self, X: torch.Tensor, K):
        device = self.device
        X = X.to(device)
        ind = int(torch.argmax(torch.linalg.norm(X, 2, dim=1), dim=0))
        mu = X[ind].unsqueeze(0).cpu()
        batch_size = 128
        size_of_x = X.size(0)
        batches = size_of_x // batch_size + 1
        indsAll = [ind]
        centInds = torch.zeros(X.size(0))
        cent = 0
        print('Kmeans ++ sampling')
        dist = torch.nn.PairwiseDistance(p=2)
        with tqdm(total=K) as pbar:
            pbar.update(mu.size(0))
            while mu.size(0) < K:
                if mu.size(0) == 1:
                    Ddist = torch.zeros(size_of_x).to(device)
                    for i in range(batches):
                        Ddist[i * batch_size: (i + 1) * batch_size] = dist(
                            X[i * batch_size: (i + 1) * batch_size].to(device), mu.to(device)).ravel()
                else:
                    newD = torch.zeros(size_of_x).to(device)
                    for i in range(batches):
                        newD[i * batch_size: (i + 1) * batch_size] = dist(
                            X[i * batch_size: (i + 1) * batch_size].to(device), mu[-1].unsqueeze(0).to(device)).ravel()
                    centInds[Ddist > newD] = cent
                    Ddist[Ddist > newD] = newD[Ddist > newD]

                if sum(Ddist) == 0.0:
                    raise ValueError("sum of distance equals to 0.")
                Ddist = Ddist.ravel()
                normed_Ddist = torch.nn.functional.normalize(Ddist, p=2, dim=-1) ** 2
                customDist = stats.rv_discrete(name='custm', values=(np.arange(len(Ddist)), normed_Ddist.cpu().numpy()))
                ind = customDist.rvs(size=1)[0]
                while ind in indsAll: ind = customDist.rvs(size=1)[0]
                mu = torch.cat([mu, X[ind].unsqueeze(0).cpu()], dim=0)
                indsAll.append(ind)
                cent += 1
                pbar.update(1)
        del X
        del mu
        gc.collect()
        torch.cuda.empty_cache()
        return indsAll

    def get_badge_embeddings(self, num_classes, mini_batch_z, mini_batch_probs, multilabel, mini_batch_pseudo_probs):
        emb_dim = mini_batch_z.size(1)
        num_of_data = mini_batch_z.size(0)
        device = mini_batch_z.device
        gradient_embeddings = torch.zeros([num_of_data, emb_dim * num_classes]).to(device)
        if not multilabel:
            # Cross-Entropy Loss
            if mini_batch_pseudo_probs is None:
                idx = torch.argmax(mini_batch_probs, dim=-1, keepdim=True)
            else:
                idx = torch.argmax(mini_batch_pseudo_probs, dim=-1, keepdim=True)
            pseudo_labels = torch.zeros_like(mini_batch_probs).scatter_(dim=1, index=idx, value=1.)
            gradient_embeddings = torch.bmm((mini_batch_probs - pseudo_labels).unsqueeze(2),
                                            mini_batch_z.unsqueeze(1)).flatten(start_dim=1)
        else:
            # BCE Loss for multi-label classification
            pseudo_labels = mini_batch_probs.clone()
            pseudo_labels[pseudo_labels >= 0.5] = 1.
            pseudo_labels[pseudo_labels < 0.5] = 0.
            for i in range(num_of_data):
                gradient_embeddings[i] = (
                            (mini_batch_probs[i] - pseudo_labels[i]).unsqueeze(1) * mini_batch_z[i].unsqueeze(
                        0)).flatten()

        return gradient_embeddings

    def query(self, n, z, probs, num_classes, multilabel=False, pseudo_probs=None):
        num_of_data = probs.size(0)
        emb_dim = z.size(1)
        gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3 - 1.5
        embedding_memory = z.element_size() * z.nelement() * num_classes / 1024 ** 3
        self.mini_batch = int(embedding_memory // gpu_memory) + 1
        final_indics = []
        for batch_idx in range(self.mini_batch):
            mini_batch_size = num_of_data // self.mini_batch

            mini_batch_probs = probs[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size]
            if pseudo_probs is not None:
                mini_batch_pseudo_probs = pseudo_probs[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size]
            else:
                mini_batch_pseudo_probs = None
            mini_batch_z = z[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size]
            gradient_embeddings = self.get_badge_embeddings(num_classes, mini_batch_z, mini_batch_probs, multilabel,
                                                            mini_batch_pseudo_probs)

            if batch_idx == self.mini_batch - 1:
                indics = self.init_centers_torch(gradient_embeddings, n // self.mini_batch + n % self.mini_batch)
            else:
                indics = self.init_centers_torch(gradient_embeddings, n // self.mini_batch)

            indics = [idx + mini_batch_size * batch_idx for idx in indics]
            final_indics += indics
        # print("kmeans++ init")
        # indics = plus_plus(concat_gradient_embeddings.cpu().numpy(), n)
        assert len(final_indics) == n
        query_ids = [self.unlabeled_sample_ids[idx] for idx in final_indics]
        assert len(query_ids) == n

        return query_ids
