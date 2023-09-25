import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch

class BALD:
    def __init__(self, unlabeled_sample_ids):

        self.unlabeled_sample_ids = unlabeled_sample_ids

    def query(self, n, probs, multilabel=False):
        """

        :param n:
        :param probs: [num_mc_samples, num_of_unlabeled_data, class_num]
        :return:
        """
        if multilabel:
            pb = probs.mean(0)
            entropy1 = (pb * torch.log(pb) + (1 - pb) * torch.log(1 - pb)).mean(1)
            entropy2 = (probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).mean(2).mean(0)
        else:
            pb = probs.mean(0)
            entropy1 = (-pb * torch.log(pb)).sum(1)
            entropy2 = (-probs * torch.log(probs)).sum(2).mean(0)

        uncertainties = entropy2 - entropy1
        indics = uncertainties.sort()[1][:n].cpu().tolist()
        query_ids = [self.unlabeled_sample_ids[i] for i in indics]

        return query_ids
