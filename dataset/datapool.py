import random
from torch.utils.data import Dataset


class Datapool(Dataset):
    def __init__(self, all_ids, mode):
        self.all_ids = all_ids
        self.mode = mode
        assert len(list(set(self.all_ids))) == len(self.all_ids), "dataset has duplicated ids"
        self.unlabeled_ids = self.all_ids.copy()
        self.labeled_ids = []
        self.sample_ids = self.all_ids.copy()  # default for val and test datapool

    def initialize(self, query_budget: int):
        # query_budget is the number of labels been queried each round
        # random initialization for first batch of labels
        self.labeled_ids = self.unlabeled_ids[:query_budget]
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]

    def query_for_label(self, queried_ids: list):
        # queried_ids are generated from query strategy
        self.labeled_ids += queried_ids
        self.unlabeled_ids = [id for id in self.all_ids if id not in self.labeled_ids]
        assert len(self.labeled_ids) + len(self.unlabeled_ids) == len(self.all_ids)

    def query(self):
        # prepare unlabeled data index for label querying
        self.mode = "query"
        print("dataset for querying")
        self.sample_ids = self.unlabeled_ids

    def train(self):
        # prepare labeled queried data index for model training
        self.mode = "train"
        print("dataset for training")
        self.sample_ids = self.labeled_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        pass
