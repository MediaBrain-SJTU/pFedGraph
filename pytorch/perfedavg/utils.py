import torch

def compute_grad(model, criterion, x, y):
    logit = model(x)
    loss = criterion(logit, y)
    grads = torch.autograd.grad(loss, model.parameters())
    return grads