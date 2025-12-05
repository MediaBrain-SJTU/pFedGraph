import torch
import numpy as np
import copy
import cvxpy as cp
    
def compute_local_test_accuracy(model, dataloader, data_distribution):

    model.eval()

    toatl_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                toatl_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (toatl_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw):
    model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if i==j:
                similarity = 0
            else:
                similarity = torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
            model_similarity_matrix[i, j] = similarity
            model_similarity_matrix[j, i] = similarity
    print("model_similarity_matrix" ,model_similarity_matrix)
    return model_similarity_matrix

def update_graph_matrix_neighbor(nets_this_round, initial_global_parameters, dw):
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw)
    graph_matrix = calculate_graph_matrix(model_difference_matrix)
    print(f'Model difference: {model_difference_matrix[0]}')
    print(f'Graph matrix: {graph_matrix}')
    return graph_matrix

def calculate_graph_matrix(model_difference_matrix):
    graph_matrix = torch.zeros((model_difference_matrix.shape[0], model_difference_matrix.shape[0]))
    self_weight = 0.3
    for i in range(model_difference_matrix.shape[0]):
        weight = torch.exp(10 * model_difference_matrix[i]) 
        weight[i] = 0
        weight = (1 - self_weight) * weight / weight.sum()
        weight[i] = self_weight
        graph_matrix[i] = weight
        
    return graph_matrix


def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w, cluster_models):
    tmp_client_state_dict = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        if client_id==0:
            print(f'Aggregation weight: {aggregation_weight_vector}. Summation: {aggregation_weight_vector.sum()}')
        
        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

    for client_id in nets_this_round.keys():
        cluster_models[client_id].load_state_dict(tmp_client_state_dict[client_id])
        
def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            _, pred_label = torch.max(out.data, 1)
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()
    net.to('cpu')
    return correct / float(total)

def compute_loss(net, test_data_loader):
    net.eval()
    loss, total = 0, 0
    net.cuda()
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            out = net(x)
            loss += torch.nn.functional.cross_entropy(out, target).item()
            total += x.data.size()[0]
    net.to('cpu')
    return loss / float(total)

