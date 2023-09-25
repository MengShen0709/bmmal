import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch

class EntropySampling:
    def __init__(self, unlabeled_sample_ids, multilabel=False):

        self.unlabeled_sample_ids = unlabeled_sample_ids
        self.multilabel = multilabel

    def query(self, n, probs):
        """

        :param n:
        :param probs: [num_of_unlabeled_data, class_num]
        :return:
        """
        if self.multilabel:
            uncertainties = (probs * torch.log(probs) + (1 - probs) * torch.log(1 - probs)).mean(1)
            indics = uncertainties.sort()[1][:n].cpu().tolist() # descending = False
            query_ids = [self.unlabeled_sample_ids[i] for i in indics]
        else:
            log_probs = torch.log(probs)
            uncertainties = (probs * log_probs).sum(1)
            indics = uncertainties.sort()[1][:n].cpu().tolist() # descending = False
            query_ids = [self.unlabeled_sample_ids[i] for i in indics]


        return query_ids
