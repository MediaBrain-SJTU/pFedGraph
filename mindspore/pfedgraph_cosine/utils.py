import copy
import numpy as np
import cvxpy as cp
import mindspore
from mindspore import Tensor, ops
import mindspore.nn as nn

# 与 test.py 一致，预先定义按类别维度取 argmax 的算子，避免 axis 关键字不兼容
argmax_op_axis1 = ops.Argmax(axis=1)


def get_param_dict(net):
    """从 mindspore 网络提取 {name: Tensor} 参数字典。"""
    param_dict = {}
    for param in net.get_parameters():
        param_dict[param.name] = ops.stop_gradient(param.data)
    return param_dict


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    """
    计算每个本地模型与全局模型差分向量的余弦距离矩阵。
    nets_this_round: {client_id: net}
    initial_global_parameters: {name: Tensor}
    dw: {client_id: {name: Tensor}} 差分缓冲
    """
    model_similarity_matrix = np.zeros((len(nets_this_round), len(nets_this_round)), dtype=np.float32)
    index_clientid = list(nets_this_round.keys())

    # 计算每个 client 的参数差分
    for i in range(len(nets_this_round)):
        model_i = get_param_dict(nets_this_round[index_clientid[i]])
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]

    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = -cosine_similarity(weight_flatten_all(dw[index_clientid[i]]),
                                          weight_flatten_all(dw[index_clientid[j]]))
            elif similarity_matric == "fc":
                diff = -cosine_similarity(weight_flatten(dw[index_clientid[i]]),
                                          weight_flatten(dw[index_clientid[j]]))
            else:
                diff = 0.0
            if diff < -0.9:
                diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
    return model_similarity_matrix


def cosine_similarity(x, y):
    """x, y: 1D Tensor，返回标量余弦相似度（mindspore.Tensor）。"""
    x_flat = x.reshape(-1).astype(mindspore.float32)
    y_flat = y.reshape(-1).astype(mindspore.float32)
    # 当前 MindSpore 版本的 ops.dot 要求输入维度 >= 2，这里改为逐元素乘法再求和
    num = ops.reduce_sum(x_flat * y_flat)
    den = (
        ops.sqrt(ops.reduce_sum(x_flat * x_flat))
        * ops.sqrt(ops.reduce_sum(y_flat * y_flat))
        + 1e-12
    )
    return num / den


def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1,
                                 similarity_matric):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = cal_model_cosine_difference(
        nets_this_round, initial_global_parameters, dw, similarity_matric
    )
    graph_matrix = optimizing_graph_matrix_neighbor(
        graph_matrix, index_clientid, model_difference_matrix, lambda_1, fed_avg_freqs
    )
    return graph_matrix


def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = -np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, P) + q.T @ x),
            [G @ x <= h, A @ x == b],
                  )
        prob.solve()
        # 将优化结果写入图矩阵（仍使用 numpy 存储）
        graph_matrix[index_clientid[i], index_clientid] = np.array(x.value, dtype=np.float32)
    return graph_matrix
  

def weight_flatten(model_dict):
    params = []
    for k, v in model_dict.items():
        if "fc" in k:
            params.append(v.reshape(-1))
    if not params:
        return Tensor(np.array([], dtype=np.float32))
    return ops.concat(params)


def weight_flatten_all(model_dict):
    params = [v.reshape(-1) for v in model_dict.values()]
    if not params:
        return Tensor(np.array([], dtype=np.float32))
    return ops.concat(params)


def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    """
    根据协作图对模型进行聚合。
    graph_matrix: numpy 数组，[client, neighbor] 的权重
    global_w: {name: Tensor}
    返回 cluster_model_vectors: {client_id: 1D Tensor}
    """
    tmp_client_state_dict = {}
    cluster_model_vectors = {}

    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = ops.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = ops.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        # 加权聚合邻居模型参数
        for neighbor_id in nets_this_round.keys():
            neighbor_params = get_param_dict(nets_this_round[neighbor_id])
            # 将 numpy.float32 转成 Python float，避免 Tensor * np.float32 的类型错误
            w = float(aggregation_weight_vector[neighbor_id])
            for key in tmp_client_state:
                tmp_client_state[key] = tmp_client_state[key] + neighbor_params[key] * w

        # 聚合成一维向量形式
        for neighbor_id in nets_this_round.keys():
            neighbor_params = get_param_dict(nets_this_round[neighbor_id])
            net_para_vec = weight_flatten_all(neighbor_params)
            # 同样避免对 1D 向量使用 ops.dot，改为逐元素乘法 + reduce_sum
            norm = ops.sqrt(ops.reduce_sum(net_para_vec * net_para_vec)) + 1e-12
            w = float(aggregation_weight_vector[neighbor_id])
            cluster_model_state = cluster_model_state + net_para_vec * (w / norm)
               
        cluster_model_vectors[client_id] = cluster_model_state

    # 将新的参数写回各自 client 模型
    for client_id in nets_this_round.keys():
        net = nets_this_round[client_id]
        params = []
        for param in net.get_parameters():
            if param.name in tmp_client_state_dict[client_id]:
                param.set_data(tmp_client_state_dict[client_id][param.name])
            params.append(param)
        net.parameters_tuple = mindspore.ParameterTuple(params)
    
    return cluster_model_vectors


def compute_acc(net, test_data_loader):
    net.set_train(False)
    correct, total = 0, 0
    for _, (x, target) in enumerate(test_data_loader.create_tuple_iterator()):
        if not isinstance(x, Tensor):
            x = Tensor(x, mindspore.float32)
        if not isinstance(target, Tensor):
            target = Tensor(target, mindspore.int32)
        out = net(x)
        # 使用预定义的 argmax 算子，等价于按类别维度取最大值
        pred_label = argmax_op_axis1(out)
        total += x.shape[0]
        correct += (pred_label == target).sum().asnumpy().item()
    return correct / float(total)


def compute_loss(net, test_data_loader):
    net.set_train(False)
    loss_total, total = 0.0, 0
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    for _, (x, target) in enumerate(test_data_loader.create_tuple_iterator()):
        if not isinstance(x, Tensor):
            x = Tensor(x, mindspore.float32)
        if not isinstance(target, Tensor):
            target = Tensor(target, mindspore.int32)
        out = net(x)
        loss = loss_fn(out, target)
        loss_total += float(loss.asnumpy())
        total += x.shape[0]
    return loss_total / float(total)

