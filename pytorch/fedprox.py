import random
import copy
import time
import math
import numpy as np
import torch
import torch.optim as optim
from fedavg.config import get_args
from model import simplecnn, textcnn
from test import compute_local_test_accuracy, compute_acc, evaluate_global_model
from prepare_data import get_dataloader
from attack import *

def local_train_fedavg(args, nets_this_round, global_model, train_local_dls):
    global_model.cuda()
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]

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
            for param_p, param in zip(global_model.parameters(), net.parameters()):
                loss += ((args.lambda_1 / 2) * torch.norm((param - param_p)) ** 2)
            loss.backward()
            optimizer.step()
        net.to('cpu')
    global_model.to('cpu')

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
global_parameters = global_model.state_dict()
local_models = []
best_val_acc_list, best_test_acc_list = [],[]

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)
    

for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    global_w = global_model.state_dict()        # Global Model Initialization

    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.load_state_dict(global_w)
    
    # Local Model Training
    local_train_fedavg(args, nets_this_round, global_model, train_local_dls)
    # Aggregation Weight Calculation
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    if round==0 or args.sample_fraction<1.0:
        print(f'Dataset size weight : {fed_avg_freqs}')
    
    manipulate_gradient(args, global_model, nets_this_round, benign_client_list)

    # Model Aggregation
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]

    global_model.load_state_dict(global_w)          # Update the global model
    mean_personalized_acc = evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)

 