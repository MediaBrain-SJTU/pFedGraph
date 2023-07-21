import random
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cal_model_difference(nets_this_round, nets_param_start, dW):
    model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        model_i_start = nets_param_start[index_clientid[i]].state_dict()
        for key in dW[index_clientid[i]]:
            dW[index_clientid[i]][key] =  model_i[key] - model_i_start[key]
        nets_this_round[index_clientid[i]].load_state_dict(model_i_start)
    for i in range(len(nets_this_round)):
        for j in range(i+1, len(nets_this_round)):
            sim = torch.nn.functional.cosine_similarity(weight_flatten(dW[index_clientid[i]]).unsqueeze(0), weight_flatten(dW[index_clientid[j]]).unsqueeze(0))
            model_similarity_matrix[i, j] = sim
            model_similarity_matrix[j, i] = sim                                             
    return model_similarity_matrix

def compute_max_update_norm(cluster):
    return np.max([torch.norm(weight_flatten(client_dw)).item() for client_dw in cluster])

def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([weight_flatten(client_dw) for client_dw in cluster]), dim=0)).item()

def weight_flatten(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def cluster_clients(S):
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)

    c1 = np.argwhere(clustering.labels_ == 0).flatten() 
    c2 = np.argwhere(clustering.labels_ == 1).flatten() 
    return c1, c2


def reduce_add_average(targets, sources):
    for target in targets:
        
        for k, v in target.named_parameters():
            tmp = torch.mean(torch.stack([source[k].data for source in sources]), dim=0).clone()
            v.data += tmp
