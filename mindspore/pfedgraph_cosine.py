import copy
import random
import time
import os
import logging
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import Tensor, ops

from test import compute_acc, compute_local_test_accuracy
from pfedgraph_cosine.config import get_args
from pfedgraph_cosine.utils import (
    aggregation_by_graph,
    update_graph_matrix_neighbor,
    get_param_dict,
)
from model import simplecnn
from prepare_data import get_dataloader

# 设置日志目录
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志记录器
logging.basicConfig(
    filename=os.path.join(log_dir, f"pfedgraph_cosine_train_{time.strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,  # 记录所有INFO级别以上的日志
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def _flatten_model_params(net):
    """将网络所有参数展平为一维向量（保持梯度）。"""
    params = [p.reshape(-1) for p in net.get_parameters()]
    if not params:
        return Tensor(np.array([], dtype=np.float32))
    return ops.concat(params)


def local_train_pfedgraph(
    args,
    round,
    nets_this_round,
    cluster_models,
    train_local_dls,
    val_local_dls,
    test_dl,
    data_distributions,
    best_val_acc_list,
    best_test_acc_list,
    benign_client_list,
):
    """
    使用与当前 fedavg.py 相同风格的 MindSpore 训练逻辑：
      - 对每个 client 单独创建 optimizer 和 loss_fn
      - 定义 forward_fn(data, label)：返回标量 loss（交叉熵 + 图正则项）
      - 使用 mindspore.value_and_grad(forward_fn, None, optimizer.parameters)
      - 训练循环中：loss, grads = grad_fn(x, target); optimizer(grads)
    """

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(
                net, test_dl, data_distribution
            )

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print(
                ">> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}".format(
                    net_id, personalized_test_acc, generalized_test_acc
                )
            )
            logger.info(
                f">> Client {net_id} test1 | (Pre) Personalized Test Acc: ({personalized_test_acc}) | Generalized Test Acc: {generalized_test_acc}"
            )

        # Set Optimizer（与 fedavg 一致，每个 client 单独一个 optimizer）
        params = net.trainable_params()
        if args.optimizer == "adam":
            optimizer = nn.Adam(params, learning_rate=args.lr, weight_decay=args.reg)
        elif args.optimizer == "sgd":
            optimizer = nn.SGD(params, learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)
        else:
            optimizer = nn.Adam(params, learning_rate=args.lr, weight_decay=args.reg)

        net.set_train(True)

        # --- 与 fedavg.py 相同风格：forward_fn + value_and_grad ---
        def forward_fn(data, label):
            """
            计算当前 client 的损失：
              - 基础交叉熵 ce_loss
              - 若 round > 0 且该 client 在 cluster_models 中，则加上图聚类正则项
            """
            logits = net(data)
            ce_loss = loss_fn(logits, label)

            if round > 0 and net_id in cluster_models:
                flat_params = _flatten_model_params(net)
                cluster_vec = cluster_models[net_id]
                if not isinstance(cluster_vec, Tensor):
                    cluster_vec = Tensor(cluster_vec, mindspore.float32)
                # 当前 MindSpore 版本对 1D Tensor 不支持 ops.dot，这里改为逐元素乘法 + reduce_sum
                num = ops.reduce_sum(cluster_vec * flat_params)
                den = ops.sqrt(ops.reduce_sum(flat_params * flat_params)) + 1e-12
                loss_reg = args.lam * num / den
                return ce_loss + loss_reg

            return ce_loss

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

        iterator = train_local_dl.create_tuple_iterator()
        last_loss = None
        for _ in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = train_local_dl.create_tuple_iterator()
                x, target = next(iterator)

            if not isinstance(x, Tensor):
                x = Tensor(x, mindspore.float32)
            if not isinstance(target, Tensor):
                target = Tensor(target, mindspore.int32)

            loss, grads = grad_fn(x, target)
            optimizer(grads)
            last_loss = loss

        if last_loss is not None:
            logger.info(f"[pfedgraph] Client {net_id} last local loss: {float(last_loss.asnumpy())}")
        
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(
                net, test_dl, data_distribution
            )

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print(
                ">> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}".format(
                    net_id, personalized_test_acc, generalized_test_acc
                )
            )
            logger.info(
                f">> Client {net_id} test2 | (Pre) Personalized Test Acc: ({personalized_test_acc}) | Generalized Test Acc: {generalized_test_acc}"
            )

    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
mindspore.set_seed(seed)
random.seed(seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for _ in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for _ in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1 - args.attack_ratio)))
benign_client_list.sort()
print(f">> -------- Benign clients: {benign_client_list} --------")

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(
    args
)

model = simplecnn
    
global_model = model(cfg["classes_size"])
global_parameters = get_param_dict(global_model)
local_models = []
best_val_acc_list, best_test_acc_list = [], []
dw = []
for i in range(cfg["client_num"]):
    local_models.append(model(cfg["classes_size"]))
    param_dict = get_param_dict(local_models[i])
    dw.append({key: ops.zeros_like(value) for key, value in param_dict.items()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

# 协作图（numpy 存储，行列为 client id）
graph_matrix = np.ones((len(local_models), len(local_models)), dtype=np.float32) / (
    len(local_models) - 1
)
np.fill_diagonal(graph_matrix, 0.0)
    
cluster_model_vectors = {}
for round in range(cfg["comm_round"]):
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f">> Clients in this round : {party_list_this_round}")
    nets_this_round = {k: local_models[k] for k in party_list_this_round}

    mean_personalized_acc = local_train_pfedgraph(
        args,
        round,
        nets_this_round,
        cluster_model_vectors,
        train_local_dls,
        val_local_dls,
        test_dl,
        data_distributions,
        best_val_acc_list,
        best_test_acc_list,
        benign_client_list,
    )
   
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

    graph_matrix = update_graph_matrix_neighbor(
        graph_matrix,
        nets_this_round,
        global_parameters,
        dw,
        fed_avg_freqs,
        args.alpha,
        args.difference_measure,
    )  # Graph Matrix is not normalized yet
    cluster_model_vectors = aggregation_by_graph(
        cfg, graph_matrix, nets_this_round, global_parameters
    )  # Aggregation weight is normalized here

    print(">> (Current) Round {} | Local Per: {:.5f}".format(round, mean_personalized_acc))
    print("-" * 80)
    logger.info(f">> (Current) Round {round} | Local Per: {mean_personalized_acc}")
    logger.info("-" * 80)

print(">> Finished training")
# 保存每个客户端的模型
models_dir = "./models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
for net_id, net in nets_this_round.items():
    # 使用 MindSpore 提供的 API 保存权重
    mindspore.save_checkpoint(net, os.path.join(models_dir, f"client_{net_id}.ckpt"))