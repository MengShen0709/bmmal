# Original implementation from https://github.com/AminParvaneh/alpha_mix_active_learning
# https://arxiv.org/pdf/2006.10219.pdf CVPR 2021

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
import math
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from .k_center_greedy import KCenterGreedy

class GCNSampling:
    def __init__(self, unlabeled_sample_ids, all_sample_ids, device, method="UncertainGCN"):
        self.unlabeled_sample_ids = unlabeled_sample_ids
        self.all_sample_ids = all_sample_ids
        self.method = method
        assert self.method in ["UncertainGCN", "CoreGCN"]
        self.hidden_units = 128
        self.dropout_rate = 0.3
        self.LR_GCN = 1e-3
        self.WDECAY = 5e-4
        self.lambda_loss = 1.2
        self.s_margin = 0.1
        self.subset_size = 10000
        self.device = device

    def query(self, n, features):

        features = nn.functional.normalize(features.to(self.device))
        size_of_data = features.size(0)
        mini_batch = 20000
        scores = []
        feat = []
        for step in range(0, size_of_data, mini_batch):

            mini_batch_feature = features[step: step + mini_batch]

            adj = aff_to_adj(mini_batch_feature)

            gcn_model = GCN(nfeat=mini_batch_feature.shape[1],
                            nhid=self.hidden_units,
                            nclass=1,
                            dropout=self.dropout_rate).to(self.device)

            optim_gcn = optim.Adam(gcn_model.parameters(), lr=self.LR_GCN, weight_decay=self.WDECAY)

            mini_batch_ids = list(range(step, min(step + mini_batch, size_of_data)))

            nlbl = np.array([i for i in mini_batch_ids if self.all_sample_ids[i] in self.unlabeled_sample_ids]) - step
            lbl = np.array([i for i in mini_batch_ids if self.all_sample_ids[i] not in self.unlabeled_sample_ids]) - step

            print('Learning Graph Convolution Network...')
            gcn_model.train()
            for _ in tqdm(range(200)):
                optim_gcn.zero_grad(set_to_none=True)
                outputs, _, _ = gcn_model(mini_batch_feature, adj)
                loss = BCEAdjLoss(outputs, lbl, nlbl, self.lambda_loss)
                loss.backward()
                optim_gcn.step()

            gcn_model.eval()
            with torch.no_grad():
                with torch.cuda.device(self.device):
                    inputs = mini_batch_feature.cuda()
                    #labels = binary_labels.cuda()
                mini_batch_scores, _, mini_batch_feat = gcn_model(inputs, adj)
            scores.append(mini_batch_scores)
            feat.append(mini_batch_feat)

        scores = torch.cat(scores, dim=0)
        feat = torch.cat(feat, dim=0).detach().cpu()

        if self.method == "CoreGCN":
            sampling_method = KCenterGreedy(self.unlabeled_sample_ids, self.all_sample_ids)
            query_ids = sampling_method.query(n, feat)
        else:
            s_margin = self.s_margin
            nlbl = np.array([i for i in range(len(self.all_sample_ids)) if self.all_sample_ids[i] in self.unlabeled_sample_ids])
            scores_median = np.squeeze(torch.abs(scores[nlbl] - s_margin).detach().cpu().numpy())
            chosen = np.argsort(-(scores_median))[-n:]
            query_ids = [self.unlabeled_sample_ids[i] for i in chosen]

        del gcn_model, optim_gcn, feat, features
        torch.cuda.empty_cache()

        assert len(list(set(query_ids))) == n
        for query_id in query_ids:
            assert query_id in self.unlabeled_sample_ids

        return query_ids


class GCNDataset(Dataset):
    def __init__(self, features, adj, labeled):
        self.features = features
        self.labeled = labeled
        self.adj = adj

    def __getitem__(self, index):
        return self.features[index], self.adj[index], self.labeled[index]

    def __len__(self):
        return len(self.features)


def aff_to_adj(x):
    x = x.detach()
    adj = torch.matmul(x, x.T)
    adj += -1.0 * torch.eye(adj.size(0)).to(x.device)
    adj_diag = torch.sum(adj, dim=0)
    adj = torch.matmul(adj, torch.diag(1/adj_diag).to(x.device))
    adj = adj + torch.eye(adj.size(0)).to(x.device)
    return adj


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(feat, adj)
        #x = self.linear(x)
        # x = F.softmax(x, dim=1)
        return torch.sigmoid(x), feat, torch.cat((feat,x),1)

