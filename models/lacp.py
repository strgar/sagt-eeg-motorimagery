import torch
import torch.nn as nn


class LearnableLACP(nn.Module):
"""Learnable Adaptive Connectivity Prior"""
def __init__(self, num_nodes: int):
super().__init__()
self.B = nn.Parameter(torch.zeros(num_nodes, num_nodes))


def forward(self):
return self.B
