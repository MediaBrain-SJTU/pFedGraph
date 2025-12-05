import random
import time
import numpy as np
import tensorflow as tf
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

# 设置日志记录器
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
        if args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=args.reg, amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)
        else:
             optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr) # fallback

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            # TF 自动处理 GPU，无需手动 cuda()
            with tf.GradientTape() as tape:
                out = net(x, training=True)
                loss = loss_fn(target, out)
            
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

args, cfg = get_args()
print(args)
logger.info(f"Arguments: {args}")
seed = args.init_seed
np.random.seed(seed)
tf.random.set_seed(seed)
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

# 初始化模型
# TF Keras 需要 build 或运行一次才能确定权重形状
dummy_model = simplecnn(cfg['classes_size'])
dummy_input = tf.zeros((1, 32, 32, 3)) # 假设 CIFAR
dummy_model(dummy_input)

global_parameters = dummy_model.get_weights() # List of numpy arrays
local_models = []
best_val_acc_list, best_test_acc_list = [],[]

for i in range(args.n_parties):
    m = simplecnn(cfg['classes_size'])
    m(dummy_input) # build
    local_models.append(m)
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)
    
# 将 global_parameters 定义为 numpy 列表用于 set_weights
global_w = [w.copy() for w in global_parameters]

for round in range(args.comm_round):          # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction<1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    
    # Global Model Initialization (distribution)
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        net.set_weights(global_w)
    
    # Local Model Training
    local_train_fedavg(args, nets_this_round, train_local_dls)
    
    # Aggregation Weight Calculation
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    if round==0 or args.sample_fraction<1.0:
        print(f'Dataset size weight : {fed_avg_freqs}')

    # Model Aggregation (FedAvg)
    # global_w 是 list of numpy arrays
    new_global_w = []
    
    # 假设所有模型结构一致
    model_0_weights = nets_this_round[party_list_this_round[0]].get_weights()
    for layer_idx in range(len(model_0_weights)):
        # 聚合该层
        weighted_layer = np.zeros_like(model_0_weights[layer_idx])
        for i, net_id in enumerate(party_list_this_round):
            net_weights = nets_this_round[net_id].get_weights()
            weighted_layer += net_weights[layer_idx] * fed_avg_freqs[i]
        new_global_w.append(weighted_layer)
        
    global_w = new_global_w
    dummy_model.set_weights(global_w) # Update global model object for eval
    
    mean_personalized_acc = evaluate_global_model(args, nets_this_round, dummy_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)
    logger.info(f">> (Current) Round {round} | Local Per: {mean_personalized_acc}")
    logger.info('-'*80)