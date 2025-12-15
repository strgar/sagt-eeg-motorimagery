import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.lacp import LearnableLACP


class SAGTLayer(nn.Module):
def __init__(self, dim: int, num_nodes: int):
super().__init__()
self.q = nn.Linear(dim, dim)
self.k = nn.Linear(dim, dim)
self.v = nn.Linear(dim, dim)
self.lacp = LearnableLACP(num_nodes)


def forward(self, x, A_plv):
Q = self.q(x)
K = self.k(x)
V = self.v(x)


attn = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(x.size(-1))
attn = attn + A_plv + self.lacp()
attn = F.softmax(attn, dim=-1)
return torch.matmul(attn, V)




class GraphTransformer(nn.Module):
def __init__(self, in_dim: int, hidden_dim: int, num_nodes: int, num_classes: int):
super().__init__()
self.embed = nn.Linear(in_dim, hidden_dim)
self.sagt = SAGTLayer(hidden_dim, num_nodes)
self.pool = nn.AdaptiveAvgPool1d(1)
self.cls = nn.Linear(hidden_dim, num_classes)


def forward(self, x, A):
x = self.embed(x)
x = self.sagt(x, A)
x = self.pool(x.transpose(1, 2)).squeeze(-1)
return self.cls(x)
