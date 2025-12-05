import copy
import random
from test import compute_acc, compute_local_test_accuracy
import time
import numpy as np
import tensorflow as tf
from pfedgraph_cosine.config import get_args
from pfedgraph_cosine.utils import aggregation_by_graph, update_graph_matrix_neighbor, get_model_map
from model import simplecnn
from prepare_data import get_dataloader

import os
import logging

# 设置日志目录
log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志记录器
logging.basicConfig(
    filename=os.path.join(log_dir, f'pfedgraph_cosine_train_{time.strftime("%Y%m%d_%H%M%S")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
            logger.info(f">> Client {net_id} test1 | (Pre) Personalized Test Acc: ({personalized_test_acc}) | Generalized Test Acc: {generalized_test_acc}")

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, weight_decay=args.reg, amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        if round > 0:
            cluster_model = cluster_models[net_id] # This is a tensor
        
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            
            with tf.GradientTape() as tape:
                out = net(x, training=True)
                loss = loss_fn(target, out)

                if round > 0:
                    # 获取当前模型的所有变量并展平
                    flatten_model_list = []
                    # 确保按顺序 (util.py 逻辑)
                    model_map = get_model_map(net)
                    for k in sorted(model_map.keys()):
                        flatten_model_list.append(tf.reshape(model_map[k], [-1]))
                    
                    if flatten_model_list:
                        flatten_model = tf.concat(flatten_model_list, axis=0)
                        
                        # loss2 = args.lam * dot(cluster, model) / norm(model)
                        dot_prod = tf.tensordot(cluster_model, flatten_model, axes=1)
                        norm_model = tf.norm(flatten_model)
                        
                        if norm_model > 0:
                            loss2 = args.lam * dot_prod / norm_model
                        else:
                            loss2 = 0.0
                        
                        loss += loss2

            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
            logger.info(f">> Client {net_id} test2 | (Pre) Personalized Test Acc: ({personalized_test_acc}) | Generalized Test Acc: {generalized_test_acc}")
    
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


args, cfg = get_args()
print(args)
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
dummy_model = simplecnn(cfg['classes_size'])
dummy_input = tf.zeros((1, 32, 32, 3))
dummy_model(dummy_input)

# === 修正 global_parameters_dict 的构造方式 ===
# 之前做法：
#   global_param_values = dummy_model.get_weights()
#   global_model_vars = dummy_model.trainable_variables
#   global_parameters_dict = {v.name: tf.convert_to_tensor(val)
#                             for v, val in zip(global_model_vars, global_param_values)}
# 这里假设 trainable_variables 和 get_weights() 的顺序完全一致，
# 但在 Keras 中，两者顺序并不保证严格一一对应，可能导致
#   某个变量名（例如 conv2d kernel）的 key 对应到 fc3 的权重 (84, 10)，
# 后面在 utils.aggregation_by_graph 中按名字回写时，就会出现
#   conv kernel 形状 (5, 5, 3, 6) 却拿到 (84, 10) 的值，触发 shape mismatch。
#
# 更安全的做法：直接用变量本身当前的值来构造
# {index(int) -> tensor} 的字典，避免通过名字配对，
# 并与 utils.get_model_map 使用的命名方式保持一致。
from pfedgraph_cosine.utils import get_model_map
global_parameters_dict = {k: tf.identity(v) for k, v in get_model_map(dummy_model).items()}

# 同时仍然保留一份初始权重列表，用于将 dummy_model 的初始参数拷贝到各个本地模型中
global_param_values = dummy_model.get_weights()

local_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []

for i in range(cfg['client_num']):
    m = simplecnn(cfg['classes_size'])
    m(dummy_input)  # build
    local_models.append(m)
    # dw structure: dict of index -> tensor（具体内容会在 cal_model_cosine_difference 中被覆盖）
    dw.append({idx: tf.zeros_like(v) for idx, v in enumerate(m.trainable_variables)})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

# Graph Matrix (TensorFlow Tensor)
graph_matrix = tf.ones((len(local_models), len(local_models))) / (len(local_models)-1)
# Set diagonal to 0
diag_mask = 1 - tf.eye(len(local_models))
graph_matrix = graph_matrix * diag_mask

# Load global to local
for net in local_models:
    net.set_weights(global_param_values)

cluster_model_vectors = {}

for round in range(cfg["comm_round"]):
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
        
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    
    # 记录起始参数 (用于计算 dw) —— 当前实现中未使用，如后续需要可改为 index 方式
    nets_param_start_dict = {}
    for k in party_list_this_round:
        w_vars = local_models[k].trainable_variables
        nets_param_start_dict[k] = {idx: tf.identity(v) for idx, v in enumerate(w_vars)}

    mean_personalized_acc = local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list)
   
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

    # 更新 Graph Matrix
    # global_parameters_dict 在这里充当 reference，但在 aggregation_by_graph 中会被更新吗？
    # 原逻辑中 global_w 在 aggregation_by_graph 中并没有被显式更新回 global_model，而是作为初始值
    # 这里我们需要传入当前 global 的状态
    
    graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters_dict, dw, fed_avg_freqs, args.alpha, args.difference_measure)
    
    # 聚合
    cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters_dict)
    
    # 更新 global_parameters_dict 为下一轮做准备 (虽然 pFedGraph 主要是个性化，但可能需要维护一个全局状态概念)
    # 在原代码中，global_parameters 似乎是静态的初始值？
    # 查看原代码: global_parameters = global_model.state_dict() 在循环外初始化。
    # 在循环内，aggregation_by_graph 使用它作为 tmp_client_state_dict 的结构模板。
    # 看起来 global_parameters 在循环中保持不变（作为初始化模板），或者作为上一轮的参考。
    # 如果作为参考，它应该在每轮更新。但在原代码 FedAvg 逻辑里更新了 global，而在 pFedGraph 里 global_parameters 似乎没有显式被赋值更新。
    # 我们保持原代码逻辑：global_parameters_dict 保持不变。

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)
    logger.info(f">> (Current) Round {round} | Local Per: {mean_personalized_acc}")
    logger.info('-'*80)

print('>> Training Finished')

# 保存每个客户端的最终模型（使用 Keras 推荐的 .keras 后缀）
models_dir = './models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

timestamp = time.strftime("%Y%m%d_%H%M%S")
for net_id, net in enumerate(local_models):
    model_path = os.path.join(
        models_dir,
        f'pfedgraph_cosine_client_{net_id}_model_{timestamp}.keras'
    )
    tf.keras.models.save_model(net, model_path)
    print(f'>> Client {net_id} model saved to {model_path}')