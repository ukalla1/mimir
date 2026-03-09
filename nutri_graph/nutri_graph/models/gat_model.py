import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATFrontEnd(nn.Module):

    def __init__(self, num_nodes, num_types, emb_dim=64, hidden=64, heads=4, dropout=0.2):
        super().__init__()

        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.type_emb = nn.Embedding(num_types, emb_dim)

        self.gat1 = GATv2Conv(
            emb_dim,
            hidden,
            heads=heads,
            dropout=dropout,
            edge_dim=1,
        )

        self.gat2 = GATv2Conv(
            hidden * heads,
            hidden,
            heads=1,
            dropout=dropout,
            edge_dim=1,
        )

        # Residual projections
        self.res1 = nn.Linear(emb_dim, hidden * heads, bias=False)
        self.res2 = nn.Linear(hidden * heads, hidden, bias=False)

        self.ln1 = nn.LayerNorm(hidden * heads)
        self.ln2 = nn.LayerNorm(hidden)

        self.drop = nn.Dropout(dropout)

        # edge existence decoder
        self.exist_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # amount decoder
        self.amount_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode(self, node_ids, node_type, edge_index, edge_attr):

        h0 = self.node_emb(node_ids) + self.type_emb(node_type)

        m1 = self.gat1(h0, edge_index, edge_attr=edge_attr)
        h1 = self.ln1(self.res1(h0) + self.drop(F.elu(m1)))

        m2 = self.gat2(h1, edge_index, edge_attr=edge_attr)
        h2 = self.ln2(self.res2(h1) + self.drop(F.elu(m2)))

        return h2

    @staticmethod
    def pair(h, edge_index):
        s, t = edge_index
        return torch.cat([h[s], h[t]], dim=-1)

    def decode_exist(self, h, edge_index):
        z = self.pair(h, edge_index)
        return self.exist_mlp(z).squeeze(-1)

    def decode_amount(self, h, edge_index):
        z = self.pair(h, edge_index)
        return self.amount_mlp(z).squeeze(-1)