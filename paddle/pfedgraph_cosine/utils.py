import paddle
import numpy as np
import copy
import cvxpy as cp


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    n_nets = len(nets_this_round)
    model_similarity_matrix = paddle.zeros((n_nets, n_nets))
    index_clientid = list(nets_this_round.keys())
    
    for i in range(n_nets):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] = model_i[key] - initial_global_parameters[key]
            
    for i in range(n_nets):
        for j in range(i, n_nets):
            if similarity_matric == "all":
                vec_i = weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0)
                vec_j = weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0)
                # paddle.nn.functional.cosine_similarity
                diff = - paddle.nn.functional.cosine_similarity(vec_i, vec_j)
                
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif similarity_matric == "fc":
                vec_i = weight_flatten(dw[index_clientid[i]]).unsqueeze(0)
                vec_j = weight_flatten(dw[index_clientid[j]]).unsqueeze(0)
                diff = - paddle.nn.functional.cosine_similarity(vec_i, vec_j)
                
                if diff < -0.9:
                    diff = -1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    return model_similarity_matrix

def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1, similarity_matric):
    index_clientid = list(nets_this_round.keys())
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lambda_1, fed_avg_freqs)
    return graph_matrix

def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()
        
        # 将 numpy 结果转回 Tensor 放入 graph_matrix
        graph_matrix[index_clientid[i], index_clientid] = paddle.to_tensor(x.value, dtype='float32')
        
    return graph_matrix
  
def weight_flatten(model):
    params = []
    for k in model:
        if 'fc' in k:
            params.append(paddle.reshape(model[k], [-1]))
    if not params:
        return paddle.to_tensor([])
    params = paddle.concat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(paddle.reshape(model[k], [-1]))
    if not params:
        return paddle.to_tensor([])
    params = paddle.concat(params)
    return params

def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    
    # 初始化
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        # zeros_like
        cluster_model_vectors[client_id] = paddle.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = paddle.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        for neighbor_id in nets_this_round.keys():
            net_para = weight_flatten_all(nets_this_round[neighbor_id].state_dict())
            # paddle.norm
            norm_val = paddle.norm(net_para)
            if norm_val > 0:
                cluster_model_state += net_para * (aggregation_weight_vector[neighbor_id] / norm_val)
               
    for client_id in nets_this_round.keys():
        # load_state_dict -> set_state_dict
        nets_this_round[client_id].set_state_dict(tmp_client_state_dict[client_id])
    
    return cluster_model_vectors


def compute_loss(net, test_data_loader):
    net.eval()
    loss_sum, total = 0, 0
    with paddle.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            target = paddle.cast(target, 'int64')
            out = net(x)
            # paddle.nn.functional.cross_entropy
            loss = paddle.nn.functional.cross_entropy(out, target)
            loss_sum += loss.item() # 注意这里通常求的是 mean loss，如果要累加需要乘以 batch size
            total += x.shape[0]
    return loss_sum / float(total) # 这里的逻辑可能需要根据 Loss reduction 调整