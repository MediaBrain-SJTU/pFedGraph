import random
import time
import numpy as np
import paddle
import paddle.optimizer as optim
from fedavg.config import get_args
from model import simplecnn
from test import compute_local_test_accuracy, compute_acc, evaluate_global_model
from prepare_data import get_dataloader
import os
import logging

# 设置日志目录
log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, f'fedavg_train_{time.strftime("%Y%m%d_%H%M%S")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def local_train_fedavg(args, nets_this_round, train_local_dls):
    
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        logger.info(f"Training model {net_id} for local iterations.")

        # Set Optimizer
        # Paddle 优化器需要传入 parameters
        if args.optimizer == 'adam':
            optimizer = optim.Adam(parameters=net.parameters(), learning_rate=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            # Paddle Adam 支持 amsgrad
            optimizer = optim.Adam(parameters=net.parameters(), learning_rate=args.lr, weight_decay=args.reg, amsgrad=True)
        elif args.optimizer == 'sgd':
            # Paddle SGD 默认没有 momentum，需要手动指定
            optimizer = optim.SGD(parameters=net.parameters(), learning_rate=args.lr, weight_decay=args.reg) 
            # 注意: 如果需要 momentum=0.9，Paddle 使用的是 optim.Momentum
            if hasattr(args, 'momentum') and args.momentum > 0:
                 optimizer = optim.Momentum(parameters=net.parameters(), learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)

        criterion = paddle.nn.CrossEntropyLoss()
        
        net.train()
            
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            # Paddle 自动处理设备
            optimizer.clear_grad() # zero_grad -> clear_grad
            target = paddle.cast(target, 'int64')

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

args, cfg = get_args()
print(args)
logger.info(f"Arguments: {args}")
seed = args.init_seed
np.random.seed(seed)
paddle.seed(seed) # torch.manual_seed
random.seed(seed)

# 设置 GPU
if paddle.device.is_compiled_with_cuda():
    paddle.set_device('gpu')

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

model = simplecnn
        
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
        net.set_state_dict(global_w) # load_state_dict -> set_state_dict
    
    # Local Model Training
    local_train_fedavg(args, nets_this_round, train_local_dls)
    # Aggregation Weight Calculation
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    if round==0 or args.sample_fraction<1.0:
        print(f'Dataset size weight : {fed_avg_freqs}')

    # Model Aggregation
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]

    global_model.set_state_dict(global_w)          # Update the global model
    mean_personalized_acc = evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)
    logger.info(f">> (Current) Round {round} | Local Per: {mean_personalized_acc}")
    logger.info('-'*80)