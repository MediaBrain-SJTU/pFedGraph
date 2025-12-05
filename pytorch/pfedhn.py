import random
import copy
import time
import math
import numpy as np
import torch
import torch.optim as optim
from collections import OrderedDict, defaultdict

from pfedhn.config import get_args
from pfedhn.model import simplecnn, simplecnn_hypernetwork
from test import compute_local_test_accuracy, compute_acc
from prepare_data import partition_data, get_dataloader
from attack import *

def local_train_fedavg(args, nets_this_round, model_us, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list):

    for net_id, net in nets_this_round.items():
        net.cuda()
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
        
        val_acc = compute_acc(net, val_local_dls[net_id])
        
        personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)
        
        if val_acc > best_val_acc_list[net_id]:
            best_val_acc_list[net_id] = val_acc
            best_test_acc_list[net_id] = personalized_test_acc

        print('>> Client {} | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list).mean()


args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

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

embed_dim = args.embed_dim if args.embed_dim!=1 else int(1 + args.n_parties / 4)


if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
    
local_models = []
best_val_acc_list, best_test_acc_list = [],[]

hnet = simplecnn_hypernetwork(args.n_parties, embed_dim, out_dim=cfg['classes_size'], hidden_dim=args.hyper_hid, n_hidden_layer=args.n_hidden_layer)
hnet = hnet.cuda()
optimizer = torch.optim.SGD(params=hnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)
    

for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    
    # sample a client
    client_id = random.choice(party_list_this_round)
    local_model = local_models[client_id]
    local_model.cuda()

    # produce & load local network weights
    weights = hnet(torch.tensor([client_id], dtype=torch.long).cuda())
    local_model.load_state_dict(weights)

    # init inner optimizer
    inner_optim = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=.9, weight_decay=args.reg)

    # storing theta_i for later calculating delta theta
    inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

    # inner updates -> obtaining theta_tilda
    local_model.train()
    train_local_dl = train_local_dls[client_id]
    data_distribution = data_distributions[client_id]
    iterator = iter(train_local_dl)
    for iteration in range(args.num_local_iterations):
        try:
            x, target = next(iterator)
        except StopIteration:
            iterator = iter(train_local_dl)
            x, target = next(iterator)
        x, target = x.cuda(), target.cuda()
        target = target.long()
        
        inner_optim.zero_grad()
        optimizer.zero_grad()

        out = local_model(x)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(local_model.parameters(), 50)
        inner_optim.step()
    
    optimizer.zero_grad()

    if client_id not in benign_client_list:
        manipulate_one_model(args, local_model, client_id)

    final_state = local_model.state_dict()

    # calculating delta theta
    delta_theta = OrderedDict({k: inner_state[k] - final_state[k] for k in weights.keys()})

    # calculating phi gradient
    hnet_grads = torch.autograd.grad(
        list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
    )

    # update hnet weights
    for p, g in zip(hnet.parameters(), hnet_grads):
        p.grad = g

    torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
    optimizer.step()

    # evaluation
    val_acc = compute_acc(local_model, val_local_dls[client_id])
    personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(local_model, test_dl, data_distribution)
    
    if val_acc > best_val_acc_list[client_id]:
        best_val_acc_list[client_id] = val_acc
        best_test_acc_list[client_id] = personalized_test_acc

    print('>> Client {} | Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(client_id, personalized_test_acc, generalized_test_acc))
    local_model.to('cpu')

    mean_personalized_acc = np.array(best_test_acc_list).mean()
    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)

 