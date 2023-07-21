import random
import copy
import time
import math
import numpy as np
import torch
import torch.optim as optim
from cluster.config import get_args
from model import simplecnn, textcnn
from test import compute_local_test_accuracy, compute_acc
from prepare_data import get_dataloader
from cluster.utils import cal_model_difference, compute_max_update_norm, compute_mean_update_norm, cluster_clients, reduce_add_average
from attack import *
 
def local_train(args, nets_this_round, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
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
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


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

benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
benign_client_list.sort()
print(f'>> -------- Benign clients: {benign_client_list} --------')

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn
    
global_model = model(cfg['classes_size'])
global_w = global_model.state_dict()

local_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

for net in local_models:
    net.load_state_dict(global_w)
    
cluster_indices =  [np.arange(args.n_parties).astype("int")]
client_clusters = [[local_models[i] for i in idcs] for idcs in cluster_indices]

for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')

    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
    
    # Local Model Training
    mean_personalized_acc = local_train(args, nets_this_round, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)
    
    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)
    
    similarity = cal_model_difference(nets_this_round, nets_param_start, dw)
    print(similarity)
    cluster_indices_new = []
    for idc in cluster_indices:
        max_norm = compute_max_update_norm([dw[i] for i in idc])
        mean_norm = compute_mean_update_norm([dw[i] for i in idc])
        print(mean_norm, max_norm, args.eps1, args.eps2)
        if mean_norm < args.eps1 and max_norm > args.eps2 and len(idc) > 2:
            c1, c2 = cluster_clients(similarity[idc][:,idc])
            print("new split", idc, c1, c2)
            cluster_indices_new += [idc[c1], idc[c2]]
        else:
            cluster_indices_new += [idc]
        
    cluster_indices = cluster_indices_new
    client_clusters = [[local_models[i] for i in idcs] for idcs in cluster_indices]
    gradient_clusters = [[dw[i] for i in idcs] for idcs in cluster_indices]
    for i in range(len(cluster_indices)):
        reduce_add_average(client_clusters[i], gradient_clusters[i])

    print("cluster", cluster_indices)
    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)

 