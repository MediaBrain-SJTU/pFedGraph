import random
import copy
import time
import math
import numpy as np
import torch
import torch.optim as optim
from fedrod.config import get_args
from fedrod.utils import balanced_softmax_loss, compute_local_test_accuracy, compute_acc
from model import simplecnn, textcnn
from prepare_data import get_dataloader
from attack import *

def local_train_fedavg(args, nets_this_round, p_models, train_local_dls, val_local_dls, test_dl, traindata_cls_counts, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]
        p_model = p_models[net_id]
        sample_per_class = torch.Tensor(traindata_cls_counts[net_id])
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        poptimizer = torch.optim.SGD(p_model.parameters(), lr = args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()
        p_model.cuda()
        p_model.train()

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

            rep = net.base(x)
            out_g = net.classifier(rep)
            loss_bsm = balanced_softmax_loss(target, out_g, sample_per_class)
            loss_bsm.backward()
            optimizer.step()
            
            out_p = p_model(rep.detach())
            loss = criterion(out_g.detach() + out_p, target)
            poptimizer.zero_grad()
            loss.backward()
            poptimizer.step()

        if net_id in benign_client_list:
            val_acc = compute_acc(net, p_model, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, p_model, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
        p_model.to('cpu')
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
global_parameters = global_model.state_dict()
local_models = []
p_models = []
best_val_acc_list, best_test_acc_list = [],[]

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))
    p_models.append(copy.deepcopy(local_models[i].classifier))
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)
    

for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    global_w = global_model.base.state_dict()        # Global Model Initialization

    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.base.load_state_dict(global_w)
    
    # Local Model Training
    mean_personalized_acc = local_train_fedavg(args, nets_this_round, p_models, train_local_dls, val_local_dls, test_dl, traindata_cls_counts, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)
    # Aggregation Weight Calculation
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    if round==0 or args.sample_fraction<1.0:
        print(f'Dataset size weight : {fed_avg_freqs}')

    manipulate_gradient(args, global_model, nets_this_round, benign_client_list)

    # Model Aggregation
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.base.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]

    global_model.base.load_state_dict(global_w)          # Update the global model

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)

 