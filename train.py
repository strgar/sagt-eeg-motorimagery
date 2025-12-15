import torch
import torch.nn as nn




def train_epoch(model, loader, optimizer, device):
model.train()
criterion = nn.CrossEntropyLoss()
for X, A, y in loader:
X, A, y = X.to(device), A.to(device), y.to(device)
optimizer.zero_grad()
loss = criterion(model(X, A), y)
loss.backward()
optimizer.step()
