import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm


class KMeansSampling:
    def __init__(self, unlabeled_sample_ids):

        self.unlabeled_sample_ids = unlabeled_sample_ids

    def kmeans(self, cluster_number, feature_embeddings):
        kmeans_batch_size = 256 * 16 if cluster_number < 256 * 16 else cluster_number
        feature_embeddings = feature_embeddings.cpu().numpy()
        cluster_learner = MiniBatchKMeans(init="k-means++", n_clusters=cluster_number, batch_size=kmeans_batch_size,
                                          random_state=0)
        data_size = feature_embeddings.shape[0]
        for i in tqdm(range((data_size - 1) // kmeans_batch_size + 1), desc="MiniBatchKMeans"):
            cluster_learner.partial_fit(feature_embeddings[i * kmeans_batch_size: (i + 1) * kmeans_batch_size])

        cluster_idxs = cluster_learner.predict(feature_embeddings)  # Size [num_data_samples]

        centers = cluster_learner.cluster_centers_[cluster_idxs]  # the closet cluster center for each embedding,
        #  Size [num_data_samples, feature_size]

        # cos_dis = 1 - np.matmul(feature_embeddings, centers.T).diagonal() \
        #           / np.linalg.norm(feature_embeddings, 2, axis=1) / np.linalg.norm(centers, 2, axis=1)
        #
        # dis = cos_dis

        dis = (feature_embeddings - centers) ** 2
        dis = dis.sum(axis=1)
        return dis, cluster_idxs

    def query(self, cluster_number, query_number, feature_embeddings, outlier_threshold=None):
        """

        :param outlier_threshold: int number for pruning small cluster
        :param feature_embeddings: [num_of_unlabeled_data, embedding_size]
        :return:
        """

        dis, cluster_idxs = self.kmeans(cluster_number, feature_embeddings)
        if outlier_threshold is not None:
            cluster_point_number = np.zeros_like(cluster_idxs)
            for i in range(cluster_number):
                cluster_point_number[cluster_idxs == i] = np.sum(cluster_idxs == i)
            outlier_indices = np.where(cluster_point_number <= outlier_threshold)
            non_outlier_indices = np.where(cluster_point_number > outlier_threshold)

            outlier_indices = outlier_indices[0]
            non_outlier_indices = non_outlier_indices[0]
            if len(non_outlier_indices) > query_number:
                feature_embeddings = feature_embeddings[non_outlier_indices]
                self.unlabeled_sample_ids = [self.unlabeled_sample_ids[i] for i in non_outlier_indices]
                print(">>> removed outliers >>>", len(outlier_indices))
                dis, cluster_idxs = self.kmeans(cluster_number, feature_embeddings)

            else:
                print(">>> not removing outliers >>>")

        cluster_arraies = [np.arange(feature_embeddings.shape[0])[cluster_idxs == i][np.argsort(dis[cluster_idxs == i])[:(query_number // cluster_number + 1)]] for i in
                  range(cluster_number)]
        q_idxs = []
        for cluster_array in cluster_arraies:
            for q_idx in cluster_array.tolist():
                q_idxs.append(q_idx)
        q_idxs = q_idxs[:query_number]
        query_ids = [self.unlabeled_sample_ids[i] for i in q_idxs]

        return query_ids


