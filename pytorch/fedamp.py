import copy
import math
import random
import time
from test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from fedamp.config import get_args
from fedamp.utils import aggregation_by_graph, update_graph_matrix_neighbor
from model import simplecnn, textcnn
from prepare_data import get_dataloader


def local_train_fedavg(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list):
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]
        cluster_model = cluster_models[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        
        cluster_model.cuda()
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            
            if round > 0:
                for param_p, param in zip(cluster_model.parameters(), net.parameters()):
                    loss += ((args.lambda_1 / 2) * torch.norm((param - param_p)) ** 2)
                
            loss.backward()
            optimizer.step()
        
        val_acc = compute_acc(net, val_local_dls[net_id])
        
        personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

        if val_acc > best_val_acc_list[net_id]:
            best_val_acc_list[net_id] = val_acc
            best_test_acc_list[net_id] = personalized_test_acc
        print('>> Client {} test 2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
        cluster_model.to('cpu')
        
    return np.array(best_test_acc_list).mean()


args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn
    
global_model = model(cfg['classes_size'])
global_parameters = global_model.state_dict()
local_models = []
cluster_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
for i in range(cfg['client_num']):
    local_models.append(model(cfg['classes_size']))
    cluster_models.append(model(cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)


for net in local_models:
    net.load_state_dict(global_parameters)
for net in cluster_models:   
    net.load_state_dict(global_parameters)
    
for round in range(cfg["comm_round"]):
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}

    mean_personalized_acc = local_train_fedavg(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list)
   
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

    graph_matrix = update_graph_matrix_neighbor(nets_this_round, global_parameters, dw)   # Graph Matrix is not normalized yet
    aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters, cluster_models)                                                    # Aggregation weight is normalized here

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)
 