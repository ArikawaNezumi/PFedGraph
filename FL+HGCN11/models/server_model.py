import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_add_pool, GCNConv, GINConv, GATConv, SAGEConv

from hgcn.layers.hyp_layers import GCN
from hgcn.layers.hyplayers import HgpslPool
from hgcn.layers.layers import Linear
from hgcn.layers import hyp_layers, hyplayers
from hgcn.manifolds.poincare import PoincareBall

from layers import HGPSLPool


class server_hgcn(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hgcn, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, nclass)


class server_hgcn2(nn.Module):
    def __init__(self, manifold, nfeat, nhid, nclass, nlayer, dropout, args):
        super(server_hgcn2, self).__init__()
        self.num_features = nfeat
        self.nhid = nhid
        self.args = args
        self.c = args.c
        self.manifold = PoincareBall()
        self.use_bias = args.use_bias  # 使用偏移量
        self.act = torch.nn.ReLU()
        self.pooling_ratio = 0.5
        self.sample = True
        self.sparse = True
        self.sl = True
        self.lamb = 1.0

        self.hgcn1 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, nfeat, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn2 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )
        self.hgcn3 = nn.Sequential(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, self.nhid, self.nhid, self.c, self.c, args.dropout, self.act, args.bias, args.use_att
            )
        )

        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, nclass)


class server_GCN(nn.Module):
    def __init__(self, args, num_features, nhid, num_classes):
        super(server_GCN, self).__init__()
        self.args = args
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.dropout_ratio = args.dropout

        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCN(self.nhid, self.nhid)
        self.conv3 = GCN(self.nhid, self.nhid)
        self.lin3 = torch.nn.Linear(self.nhid, self.num_classes)


class serverGIN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayer, dropout):
        super(serverGIN, self).__init__()
        self.num_layers = nlayer
        self.dropout = dropout

        self.pre = torch.nn.Sequential(torch.nn.Linear(nfeat, nhid))

        self.graph_convs = torch.nn.ModuleList()
        self.nn1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(),
                                       torch.nn.Linear(nhid, nhid))
        self.graph_convs.append(GINConv(self.nn1))
        for l in range(1):
            self.nnk = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid))
            self.graph_convs.append(GINConv(self.nnk))

        self.post1 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        # self.post2 = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(nhid, nclass))


class serverGAT(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, nlayer, dropout, is_concat=True, leaky_relu_negative_slope=0.2):
        super(serverGAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(n_feat, n_hid)
        self.conv2 = GATConv(n_hid, n_hid)
        self.conv3 = GATConv(n_hid, n_hid)
        self.post1 = torch.nn.Sequential(torch.nn.Linear(n_hid, n_hid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(n_hid, n_class))


class serverGraphSAGE(torch.nn.Module):
    def __init__(self, n_feat, n_hid, n_class, nlayer, dropout, is_concat=True, leaky_relu_negative_slope=0.2):
        super(serverGraphSAGE, self).__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(n_feat, n_hid)
        self.conv2 = SAGEConv(n_hid, n_hid)
        self.conv3 = SAGEConv(n_hid, n_hid)
        self.post1 = torch.nn.Sequential(torch.nn.Linear(n_hid, n_hid), torch.nn.ReLU())
        self.readout = torch.nn.Sequential(torch.nn.Linear(n_hid, n_class))
