import torch
import torch.nn.functional as F
from torch.nn import Linear

########################################################################################################################
from torch_geometric.nn import MLP, SAGEConv, GraphConv, ResGatedGraphConv, TransformerConv, TAGConv, ARMAConv
from torch_geometric.nn import MFConv, GCNConv
from torch_geometric.nn import ClusterGCNConv, GENConv
from torch_geometric.nn import FiLMConv, EGConv
from torch_geometric.nn import GeneralConv
########################################################################################################################
class MLPEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, act=F.mish, use_encoder_type=GENConv):
        super().__init__()
        '''
            下面的数字都是在64的基础上进行试验
                                epoch   AP          save
            SAGEConv            200     0.9365      x
            ResGatedGraphConv   200     0.8887      x
            GraphConv           200     0.6148      x
            TAGConv             200     0.9485      x
            ARMAConv            200     0.7451      x
            GeneralConv         200     0.8457      x
            TransformerConv     200     0.9262      x
            EGConv              200     0.7035      x
            ClusterGCNConv      200     0.9397      x
            FiLMConv            200     0.9440      x
            GENConv             200     0.9542      x

        '''
        self.conv1 = use_encoder_type(in_channels, out_channels)
        self.conv2 = use_encoder_type(out_channels, out_channels)
        self.conv3 = use_encoder_type(out_channels, out_channels)
        self.conv4 = use_encoder_type(out_channels, out_channels)

        self.act = act

    def forward(self, x, edge_index):
        xyxy = self.conv1(x, edge_index)
        xyxy = self.act(xyxy)
        xyxy = self.conv2(xyxy, edge_index)
        xyxy = self.act(xyxy)
        xyxy = self.conv3(xyxy, edge_index)
        xyxy = self.act(xyxy)
        output = self.conv4(xyxy, edge_index)

        return output


########################################################################################################################

class GALADecoder(torch.nn.Module):
    def __init__(self, GALA_in_channels, GALA_out_channels, act=F.leaky_relu):
        super().__init__()
        self.conv1 = GCNConv(GALA_in_channels, GALA_in_channels)
        self.conv2 = GCNConv(GALA_in_channels, 2 * GALA_out_channels)
        self.conv3 = GCNConv(2 * GALA_out_channels, GALA_out_channels)
        self.act = act

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        x = self.act(x)
        x = self.conv3(x, edge_index)
        x = self.act(x)

        return x


########################################################################################################################

class InnerProductDecoder(torch.nn.Module):
    def __init__(self, decoder_type=None):
        super().__init__()
        self.decoder_type = decoder_type
        r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
        <https://arxiv.org/abs/1611.07308>`_ paper

        .. math::
            \sigma(\mathbf{Z}\mathbf{Z}^{\top})

        where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
        space produced by the encoder."""

    def forward(self, z, edge_index):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        if self.decoder_type is not None:
            value = self.decoder_type(value)
        return value

    def forward_all(self, z):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        if self.decoder_type is not None:
            adj = self.decoder_type(adj)
        return adj

########################################################################################################################

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = self.lin1(x).relu()
        x = self.lin2(x).relu()
        return self.lin3(x)



########################################################################################################################
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from typing import Any

EPS = 1e-15


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


########################################################################################################################
class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super().__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index)
        neg_pred = self.decoder(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)