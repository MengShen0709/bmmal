import numpy as np
import torch
from tqdm import tqdm


class KCenterGreedy:

    def __init__(self, unlabeled_sample_ids, all_sample_ids, splits=1):
        self.unlabeled_sample_ids = unlabeled_sample_ids
        self.all_sample_ids = all_sample_ids
        self.min_distances = None
        self.n_obs = len(all_sample_ids)
        self.already_selected = []
        self.splits = splits

    def update_distances(self, features, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
        min_distances.
        rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = features[cluster_centers]
            if x.dim() == 1:
                x = x.unsqueeze(0)
            dist = torch.cdist(features, x, p=2)

            if self.min_distances is None:
                self.min_distances = torch.min(dist, dim=1, keepdim=True)[0]
            else:
                self.min_distances = torch.minimum(self.min_distances, dist)

    def query(self, N, features):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
        model: model with scikit-like API with decision_function implemented
        already_selected: index of datapoints already selected
        N: batch size
        Returns:
        indices of points selected to minimize distance to cluster centers
        """
        new_batch = []
        for split_index in range(self.splits):
            split_size = self.n_obs // self.splits
            if split_index == self.splits - 1:
                start_index = split_size * split_index
                end_index = split_size * split_index + split_size + self.n_obs % self.splits
                ids = list(range(start_index, end_index))
                split_N = N // self.splits + N % self.splits
            else:
                start_index = split_size * split_index
                end_index = split_size * split_index + split_size
                ids = list(range(start_index, end_index))
                split_N = N // self.splits

            already_selected = [i - start_index for i in ids if self.all_sample_ids[i] not in self.unlabeled_sample_ids]
            print(start_index, end_index)
            self.update_distances(features[start_index: end_index], already_selected, only_new=True, reset_dist=True)

            bar = tqdm(total=split_N, desc="KCG sampling")
            for _ in range(split_N):
                if self.already_selected is None:
                    # Initialize centers with a randomly selected datapoint
                    ind = int(np.random.choice(np.arange(start_index, end_index)))
                else:
                    ind = int(torch.argmax(self.min_distances)) + start_index
                    # New examples should not be in already selected since those points
                    # should have min_distance of zero to a cluster center.
                assert ind not in already_selected

                self.update_distances(features[start_index: end_index], [ind - start_index], only_new=True, reset_dist=False)
                new_batch.append(ind)
                bar.update(1)
                bar.set_description('Maximum distance from cluster centers is %0.4f' % max(self.min_distances))
                bar.display()

        # self.already_selected = already_selected

        query_ids = [self.all_sample_ids[i] for i in new_batch]
        assert len(query_ids) == N
        for id in query_ids:
            assert id in self.unlabeled_sample_ids
        return query_ids


