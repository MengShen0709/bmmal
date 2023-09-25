import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class AdversarialDeepFool:
    def __init__(self, unlabeled_sample_ids, clf, max_iter=50):
        self.unlabeled_sample_ids = unlabeled_sample_ids
        self.max_iter = max_iter
        self.clf = clf

    def cal_dis(self, z_i):
        nx = torch.unsqueeze(z_i, 0)
        nx.requires_grad_()
        eta = torch.zeros(nx.shape)
        try:
            logits, _ = self.clf(nx + eta)
        except:
            logits = self.clf(nx + eta)
        n_class = logits.topk(10, 1)[1][0, 1:].tolist()
        py = logits.max(1)[1].item()
        ny = logits.max(1)[1].item()

        i_iter = 0

        while py == ny and i_iter < self.max_iter:
            logits[0, py].backward(retain_graph=True)
            grad_np = nx.grad.data.clone()
            value_l = torch.inf
            ri = None

            for i in n_class:
                if i == py:
                    continue

                nx.grad.data.zero_()
                logits[0, i].backward(retain_graph=True)
                grad_i = nx.grad.data.clone()

                wi = grad_i - grad_np
                fi = logits[0, i] - logits[0, py]
                value_i = torch.abs(fi) / torch.linalg.norm(wi.flatten())

                if value_i < value_l:
                    ri = value_i/torch.linalg.norm(wi.flatten()) * wi

            eta += ri.clone()
            nx.grad.data.zero_()
            try:
                logits, _ = self.clf(nx + eta)
            except:
                logits = self.clf(nx + eta)
            py = logits.max(1)[1].item()
            i_iter += 1

        return (eta*eta).sum()

    def query(self, n, z):

        self.clf.cpu()
        self.clf.eval()
        dis = np.zeros(len(self.unlabeled_sample_ids))

        for i in tqdm(range(z.size(0)), ncols=100):
            # x, y, idx = unlabeled_data[i]
            dis[i] = self.cal_dis(z[i])

        query_ids = [self.unlabeled_sample_ids[idx] for idx in dis.argsort()[:n]]

        return query_ids

