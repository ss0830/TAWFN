import math
from typing import List, Optional, Tuple, Type

import torch
from torch import Tensor,nn
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class MAB(nn.Module):
    r"""Multihead-Attention Block with optional convolution and layer normalization."""

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int,
                 Conv: Optional[Type[nn.Module]] = None, layer_norm: bool = False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        # Linear layer to project Q to dim_V
        self.fc_q = Linear(dim_Q, dim_V)

        # Choose between linear or convolutional layers for K and V
        if Conv is None:
            self.fc_k = Linear(dim_K, dim_V)
            self.fc_v = Linear(dim_K, dim_V)
        else:
            self.fc_k = Conv(dim_K, dim_V)
            self.fc_v = Conv(dim_K, dim_V)

        # Layer normalization if required
        if self.layer_norm:
            self.ln0 = LayerNorm(dim_V)
            self.ln1 = LayerNorm(dim_V)

        # Output linear layer
        self.fc_o = Linear(dim_V, dim_V)

    def reset_parameters(self):
        """Resets the parameters of the layers."""
        self.fc_q.reset_parameters()
        self.fc_k.reset_parameters()
        self.fc_v.reset_parameters()
        if self.layer_norm:
            self.ln0.reset_parameters()
            self.ln1.reset_parameters()
        self.fc_o.reset_parameters()

    def forward(
            self,
            Q: Tensor,
            K: Tensor,
            graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
            mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass for the MAB block."""

        # Project Q using the linear layer
        Q_proj = self.fc_q(Q)

        # Handle graph input or standard input
        if graph is not None:
            x, edge_index, batch = graph
            K_proj = self.fc_k(x, edge_index)
            V_proj = self.fc_v(x, edge_index)
            K_proj, _ = to_dense_batch(K_proj, batch)
            V_proj, _ = to_dense_batch(V_proj, batch)
        else:
            K_proj = self.fc_k(K)
            V_proj = self.fc_v(K)

        # Split for multi-head attention
        head_dim = self.dim_V // self.num_heads
        Q_heads = torch.cat(Q_proj.split(head_dim, dim=2), dim=0)
        K_heads = torch.cat(K_proj.split(head_dim, dim=2), dim=0)
        V_heads = torch.cat(V_proj.split(head_dim, dim=2), dim=0)

        # Compute attention scores
        attention_scores = Q_heads.bmm(K_heads.transpose(1, 2)) / math.sqrt(head_dim)

        # Apply mask if provided
        if mask is not None:
            mask_repeated = mask.repeat(self.num_heads, 1, 1)
            attention_scores += mask_repeated
            attention_weights = torch.softmax(attention_scores, dim=-1)
        else:
            attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute attention output
        attention_output = attention_weights.bmm(V_heads)
        concatenated_output = torch.cat(attention_output.split(Q.size(0), dim=0), dim=2)

        # Add residual connection and apply layer normalization
        if self.layer_norm:
            concatenated_output = self.ln0(concatenated_output)

        # Final output projection with ReLU activation and optional layer normalization
        output = concatenated_output + self.fc_o(concatenated_output).relu()

        if self.layer_norm:
            output = self.ln1(output)

        return output


class PMA(torch.nn.Module):
    r"""Graph pooling with Multihead-Attention."""
    def __init__(self, channels: int, num_heads: int, num_seeds: int,
                 Conv: Optional[Type] = None, layer_norm: bool = False):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, Conv=Conv,
                       layer_norm=layer_norm)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        graph: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, graph, mask)
    
class MHA(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        Conv: Optional[Type] = None,
        num_nodes: int = 300,
        pooling_ratio: float = 0.25,
        pool_sequences: List[str] = ['GMPool_G', 'GMPool_G'],
        num_heads: int = 4,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.Conv = Conv or GCNConv
        self.num_nodes = num_nodes
        self.pooling_ratio = pooling_ratio
        self.pool_sequences = pool_sequences
        self.num_heads = num_heads
        self.layer_norm = layer_norm

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.pools = torch.nn.ModuleList()
        num_out_nodes = math.ceil(num_nodes * pooling_ratio)
        for i, pool_type in enumerate(pool_sequences):
            if pool_type not in ['GMPool_G']:
                raise ValueError("Elements in 'pool_sequences' should be  'GMPool_G'")

            if i == len(pool_sequences) - 1:
                num_out_nodes = 1

            if pool_type == 'GMPool_G':
                self.pools.append(
                    PMA(hidden_channels, num_heads, num_out_nodes,
                        Conv=self.Conv, layer_norm=layer_norm))
                num_out_nodes = math.ceil(num_out_nodes * self.pooling_ratio)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()


    def forward(self, x: Tensor, batch: Tensor,
                edge_index: Optional[Tensor] = None) -> Tensor:
        """"""
        x = self.lin1(x)
        batch_x, mask = to_dense_batch(x, batch)
        # batch_x = self.msma(batch_x)

        mask = (~mask).unsqueeze(1).to(dtype=x.dtype) * -1e9

        for i, (name, pool) in enumerate(zip(self.pool_sequences, self.pools)):
            graph = (x, edge_index, batch) if name == 'GMPool_G' else None
            batch_x = pool(batch_x, graph, mask)
            mask = None

        return self.lin2(batch_x.squeeze(1))


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, pool_sequences={self.pool_sequences})')

