import random
import time
import os
import logging
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import ops, Tensor, ParameterTuple
from fedavg.config import get_args
from model import simplecnn
from test import compute_local_test_accuracy, compute_acc, evaluate_global_model
from prepare_data import get_dataloader

# 设置日志目录
log_dir = "./logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 设置日志记录器
logging.basicConfig(
    filename=os.path.join(log_dir, f"fedavg_train_{time.strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def get_param_dict(net):
    """
    从 mindspore 网络提取一个 {name: Tensor} 的参数字典，
    方便做联邦加权平均。
    """
    param_dict = {}
    for param in net.get_parameters():
        # param.name 形如 'base.conv1.weight'
        param_dict[param.name] = ops.stop_gradient(param.data)
    return param_dict


def load_param_dict(net, param_dict):
    """
    将 {name: Tensor} 写回到网络参数中。
    """
    params = []
    for param in net.get_parameters():
        if param.name in param_dict:
            param.set_data(param_dict[param.name])
        params.append(param)
    return ParameterTuple(params)


def local_train_fedavg(args, nets_this_round, train_local_dls):
    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]
        logger.info(f"Training model {net_id} for local iterations.")

        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

        params = net.trainable_params()
        if args.optimizer == "adam":
            optimizer = nn.Adam(params, learning_rate=args.lr, weight_decay=args.reg)
        elif args.optimizer == "sgd":
            optimizer = nn.SGD(params, learning_rate=args.lr, momentum=0.9, weight_decay=args.reg)
        else:
            optimizer = nn.Adam(params, learning_rate=args.lr, weight_decay=args.reg)

        net.set_train(True)

        def forward_fn(data, label):
            logits = net(data)
            loss = loss_fn(logits, label)
            return loss

        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters)

        iterator = train_local_dl.create_tuple_iterator()
        last_loss = None
        for it in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = train_local_dl.create_tuple_iterator()
                x, target = next(iterator)

            # 确保类型正确
            if not isinstance(x, Tensor):
                x = Tensor(x, mindspore.float32)
            if not isinstance(target, Tensor):
                target = Tensor(target, mindspore.int32)

            loss, grads = grad_fn(x, target)
            optimizer(grads)
            last_loss = loss

        # 记录该 client 最后一次迭代的 loss，便于观察是否在下降
        if last_loss is not None:
            logger.info(f"Client {net_id} last local loss: {float(last_loss.asnumpy())}")


args, cfg = get_args()
print(args)
logger.info(f"Arguments: {args}")
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

for _ in range(args.n_parties):
    local_models.append(model(cfg["classes_size"]))
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)
    

for round in range(args.comm_round):  # Federated round loop
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f">> Clients in this round : {party_list_this_round}")

    # 将全局参数下发到本轮的本地模型
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    for net in nets_this_round.values():
        load_param_dict(net, global_parameters)
    
    # Local Model Training
    local_train_fedavg(args, nets_this_round, train_local_dls)

    # Aggregation Weight Calculation
    total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
    fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]
    if round == 0 or args.sample_fraction < 1.0:
        print(f"Dataset size weight : {fed_avg_freqs}")

    # Model Aggregation：对每个参数做加权平均
    new_global_params = None
    for idx, net in enumerate(nets_this_round.values()):
        net_para = get_param_dict(net)
        if new_global_params is None:
            new_global_params = {k: v * fed_avg_freqs[idx] for k, v in net_para.items()}
        else:
            for k in new_global_params:
                new_global_params[k] = new_global_params[k] + net_para[k] * fed_avg_freqs[idx]

    global_parameters = new_global_params
    load_param_dict(global_model, global_parameters)  # 更新全局模型

    mean_personalized_acc = evaluate_global_model(
        args,
        nets_this_round,
        global_model,
        val_local_dls,
        test_dl,
        data_distributions,
        best_val_acc_list,
        best_test_acc_list,
        benign_client_list,
    )

    print(">> (Current) Round {} | Local Per: {:.5f}".format(round, mean_personalized_acc))
    print("-" * 80)
    logger.info(f">> (Current) Round {round} | Local Per: {mean_personalized_acc}")
    logger.info("-" * 80)

 