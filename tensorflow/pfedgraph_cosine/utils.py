import tensorflow as tf
import numpy as np
import copy
import cvxpy as cp

def weight_flatten(model_weights_dict):
    """
    将模型权重字典展平成一维向量。
    这里的 model_weights_dict 约定为 {index(int) -> tensor}，
    我们只提取「全连接层」的参数，这里用 rank == 2 (Dense) 来近似判断，
    避免依赖不稳定的名字（如 'kernel', 'bias'）。
    """
    params = []
    # 按 index 排序以保证顺序一致性
    for k in sorted(model_weights_dict.keys()):
        w = model_weights_dict[k]
        # Dense 层的 kernel 一般是二维矩阵；卷积核通常是 4 维
        if len(w.shape) == 2:
            params.append(tf.reshape(w, [-1]))
    if not params:
        return tf.constant([])
    params = tf.concat(params, axis=0)
    return params

def weight_flatten_all(model_weights_dict):
    """
    将所有权重展平，model_weights_dict: {index(int) -> tensor}
    """
    params = []
    for k in sorted(model_weights_dict.keys()):
        params.append(tf.reshape(model_weights_dict[k], [-1]))
    if not params:
        return tf.constant([])
    params = tf.concat(params, axis=0)
    return params

def get_model_map(model):
    """
    将 Keras 模型的变量转换为 {index(int) -> tensor} 的字典。
    不再依赖变量名（如 'kernel', 'bias'），因为不同实例间名字前缀可能不同，
    甚至会发生同名冲突。Keras 对同结构模型的 trainable_variables 顺序是稳定的，
    因此用 index 可以保证跨客户端的一致性。
    """
    return {i: v for i, v in enumerate(model.trainable_variables)}

def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    n_nets = len(nets_this_round)
    model_similarity_matrix = np.zeros((n_nets, n_nets))
    index_clientid = list(nets_this_round.keys())
    
    # 计算差异 dw = model_i - global
    for i in range(n_nets):
        client_id = index_clientid[i]
        model_i_vars = get_model_map(nets_this_round[client_id])  # {idx -> var}
        
        # dw[client_id] 也使用 {idx -> tensor} 的形式
        current_dw = {}
        for key in model_i_vars:
            # 确保 key 在 initial_global 中存在
            if key in initial_global_parameters:
                current_dw[key] = model_i_vars[key] - initial_global_parameters[key]
        dw[client_id] = current_dw

    for i in range(n_nets):
        for j in range(i, n_nets):
            id_i = index_clientid[i]
            id_j = index_clientid[j]
            
            vec_i = None
            vec_j = None
            
            if similarity_matric == "all":
                vec_i = weight_flatten_all(dw[id_i])
                vec_j = weight_flatten_all(dw[id_j])
            elif similarity_matric == "fc":
                vec_i = weight_flatten(dw[id_i])
                vec_j = weight_flatten(dw[id_j])
            
            # 计算余弦相似度
            # tf.keras.losses.cosine_similarity 是 negative cosine similarity
            # 也可以手动计算: dot(a,b) / (|a|*|b|)
            norm_i = tf.norm(vec_i)
            norm_j = tf.norm(vec_j)
            
            if norm_i == 0 or norm_j == 0:
                sim = 0.0
            else:
                sim = tf.tensordot(vec_i, vec_j, axes=1) / (norm_i * norm_j)
            
            diff = -float(sim) # 保持原逻辑的负号
            
            if diff < -0.9:
                diff = -1.0
            
            model_similarity_matrix[i, j] = diff
            model_similarity_matrix[j, i] = diff

    return tf.convert_to_tensor(model_similarity_matrix, dtype=tf.float32) # 保持兼容性，但下方 cvxpy 需要 numpy

def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1, similarity_matric):
    index_clientid = list(nets_this_round.keys())
    
    # 注意: cal_model_cosine_difference 返回的是 tensor/numpy
    # 这里我们临时转回 numpy 供 cvxpy 使用
    # initial_global_parameters 这里需要是字典形式 name->value
    
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
    
    # 转换 matrix 为 numpy 以供 cvxpy
    if hasattr(model_difference_matrix, 'numpy'):
        model_difference_matrix_np = model_difference_matrix.numpy()
    else:
        model_difference_matrix_np = np.array(model_difference_matrix)

    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix_np, lambda_1, fed_avg_freqs)
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
    
    new_weights = np.zeros((n, n))
    
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector # 已经是 numpy
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()
        new_weights[i, :] = x.value

    # 更新 graph_matrix (TensorFlow tensor)
    # 保持 graph_matrix 为 tensor 方便后续计算
    indices = index_clientid
    # graph_matrix 是一个 (N, N) 的 tensor
    gm_np = graph_matrix.numpy()
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            gm_np[idx_i, idx_j] = new_weights[i, j]
            
    return tf.convert_to_tensor(gm_np, dtype=tf.float32)

def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    # global_w 是 {index(int) -> tensor} 字典
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    
    # 获取 global_w 的 flatten 形状 (作为全 0 的参考)
    flat_global = weight_flatten_all(global_w)
    
    for client_id in nets_this_round.keys():
        # 深拷贝 global_w 结构（{idx -> tensor}）
        tmp_client_state_dict[client_id] = {k: tf.identity(v) for k, v in global_w.items()}
        # 归零
        for key in tmp_client_state_dict[client_id]:
             tmp_client_state_dict[client_id][key] = tf.zeros_like(tmp_client_state_dict[client_id][key])
             
        cluster_model_vectors[client_id] = tf.zeros_like(flat_global)

    # 计算聚合
    graph_matrix_np = graph_matrix.numpy()
    
    for client_id in nets_this_round.keys():
        aggregation_weight_vector = graph_matrix_np[client_id]  # 这是一个数组
        
        # 1. 聚合参数（按 index 对齐）
        for neighbor_id, neighbor_net in nets_this_round.items():
            weight = aggregation_weight_vector[neighbor_id]
            neighbor_vars = get_model_map(neighbor_net)  # {idx -> var}
            
            for key in tmp_client_state_dict[client_id]:
                if key in neighbor_vars:
                    tmp_client_state_dict[client_id][key] += neighbor_vars[key] * weight

        # 2. 计算 cluster vector
        for neighbor_id, neighbor_net in nets_this_round.items():
            weight = aggregation_weight_vector[neighbor_id]
            neighbor_vars = get_model_map(neighbor_net)  # {idx -> var}
            flat_net = weight_flatten_all(neighbor_vars)
            
            norm_val = tf.norm(flat_net)
            if norm_val > 0:
                cluster_model_vectors[client_id] += flat_net * (weight / norm_val)

    # 应用参数回模型（按 index 对齐）
    for client_id in nets_this_round.keys():
        model = nets_this_round[client_id]
        new_weights_map = tmp_client_state_dict[client_id]  # {idx: 聚合后的 Tensor}

        for idx, v in enumerate(model.trainable_variables):
            if idx in new_weights_map:
                target_val = new_weights_map[idx]
                # ===== Debug：在赋值前检查形状是否一致 =====
                if tuple(v.shape) != tuple(target_val.shape):
                    print("=== Shape MISMATCH in aggregation_by_graph ===")
                    print(f"client_id: {client_id}")
                    print(f"var index: {idx}")
                    print(f"var name: {v.name}")
                    print(f"model var shape: {v.shape}")
                    print(f"new_weights_map[{idx}] shape: {target_val.shape}")
                    print("all idx in new_weights_map:", sorted(list(new_weights_map.keys())))
                    raise ValueError(
                        f"Shape mismatch for idx {idx}: var {tuple(v.shape)} vs new {tuple(target_val.shape)}"
                    )

                v.assign(target_val)
            # 否则保持原值
    
    return cluster_model_vectors



def compute_loss(net, test_data_loader):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_sum, total = 0, 0
    for x, target in test_data_loader:
        out = net(x, training=False)
        loss = loss_fn(target, out)
        loss_sum += loss.numpy() * x.shape[0] # accumulate sum
        total += x.shape[0]
    return loss_sum / float(total)