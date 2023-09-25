# original implementation is from https://github.com/JordanAsh/badge/blob/master/query_strategies/strategy.py
import gc

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm


class BMMAL:
    def __init__(self, unlabeled_sample_ids, device, dist="euclidean"):

        self.unlabeled_sample_ids = unlabeled_sample_ids
        self.dist = dist
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
        print('MM Balance BADGE Kmeans ++ sampling')
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

    def get_mm_badge_embeddings(self, num_classes,
                                mini_batch_z,
                                mini_batch_probs,
                                mini_batch_mm_probs,
                                mini_batch_contribution,
                                multilabel):
        emb_dims = []
        num_of_data = mini_batch_z[0].size(0)
        device = mini_batch_z[0].device
        for z in mini_batch_z:
            emb_dims.append(z.size(1))
        mm_gradient_embeddings = torch.empty([num_of_data, sum(emb_dims) * num_classes]).to(device)
        for index, (probs, z, contribution) in enumerate(zip(mini_batch_probs, mini_batch_z, mini_batch_contribution)):
            emb_dim = z.size(1)
            gradient_embeddings = torch.empty([num_of_data, emb_dim * num_classes]).to(z.device)

            if not multilabel:
                # Cross-Entropy Loss
                idx = torch.argmax(mini_batch_mm_probs, dim=-1, keepdim=True)
                pseudo_labels = torch.zeros_like(probs).scatter_(dim=1, index=idx, value=1.)
                gradient_embeddings = torch.bmm(((probs - pseudo_labels) * contribution).unsqueeze(2), z.unsqueeze(1)).flatten(
                    start_dim=1)
            else:
                # BCE Loss for multi-label classification
                pseudo_labels = probs.clone()
                pseudo_labels[pseudo_labels >= 0.5] = 1.
                pseudo_labels[pseudo_labels < 0.5] = 0.
                for i in range(num_of_data):
                    gradient_embeddings[i] = (((probs[i] - 2 * pseudo_labels[i] + probs[i] * pseudo_labels[i])
                                               * contribution[i]).unsqueeze(1) * z[i].unsqueeze(0)).flatten()

            for c in range(num_classes):
                if index == 0:
                    mm_gradient_embeddings[:,
                    c * sum(emb_dims): c * sum(emb_dims) + emb_dims[index]] = gradient_embeddings[:,
                                                                              c * emb_dim: c * emb_dim + emb_dim]
                else:
                    mm_gradient_embeddings[:,
                    c * sum(emb_dims) + emb_dims[index - 1]:c * sum(emb_dims) + emb_dims[index - 1] + emb_dims[
                        index]] = gradient_embeddings[:, c * emb_dim:c * emb_dim + emb_dim]

        return mm_gradient_embeddings

    def query(self, n, unimodal_z, unimodal_probs, unimodal_contributions, num_classes, mm_probs, multilabel=False):

        num_of_data = unimodal_z[0].size(0)
        gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 3 - 1.5
        embedding_memory = sum([z.element_size() * z.nelement() * num_classes / 1024 ** 3 for z in unimodal_z])
        self.mini_batch = int(embedding_memory // gpu_memory) + 1
        final_indics = []
        for batch_idx in range(self.mini_batch):
            mini_batch_size = num_of_data // self.mini_batch

            mini_batch_probs = [probs[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size].clone() for probs
                                in unimodal_probs]
            mini_batch_z = [z[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size].clone() for z in
                            unimodal_z]
            mini_batch_mm_probs = mm_probs[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size].clone()

            mini_batch_contribution = [contributions[batch_idx * mini_batch_size: (batch_idx + 1) * mini_batch_size].clone() for contributions
                                in unimodal_contributions]
            mm_gradient_embeddings = self.get_mm_badge_embeddings(num_classes, mini_batch_z, mini_batch_probs,
                                                                  mini_batch_mm_probs, mini_batch_contribution, multilabel)

            if batch_idx == self.mini_batch - 1:
                indics = self.init_centers_torch(mm_gradient_embeddings, n // self.mini_batch + n % self.mini_batch)
            else:
                indics = self.init_centers_torch(mm_gradient_embeddings, n // self.mini_batch)

            indics = [idx + mini_batch_size * batch_idx for idx in indics]
            final_indics += indics

        assert len(final_indics) == n
        query_ids = [self.unlabeled_sample_ids[idx] for idx in final_indics]
        assert len(query_ids) == n

        return query_ids
