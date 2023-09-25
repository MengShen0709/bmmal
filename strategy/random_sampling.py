import numpy as np
import random


class RandomSampling:
    def __init__(self, unlabeled_sample_ids):
        """

        :param dataset:
        """
        self.unlabeled_sample_ids = unlabeled_sample_ids


    def query(self, n):
        """

        :param n:
        :return: list of sample_ids
        """
        random.shuffle(self.unlabeled_sample_ids)
        sample_ids = self.unlabeled_sample_ids[:n]
        return sample_ids