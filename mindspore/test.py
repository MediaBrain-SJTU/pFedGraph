import numpy as np
import mindspore
from mindspore import ops

# 预先定义按类别维度（axis=1）取 argmax 的算子，避免关键字参数不兼容问题
argmax_op_axis1 = ops.Argmax(axis=1)

def compute_local_test_accuracy(model, dataloader, data_distribution):

    model.set_train(False)

    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    generalized_total, generalized_correct = 0, 0
    
    for batch_idx, (x, target) in enumerate(dataloader):
        x = mindspore.Tensor(x, dtype=mindspore.float32)
        target = mindspore.Tensor(target, dtype=mindspore.int32)
        out = model(x)
        # 按类别维度取最大概率对应的类别
        pred_label = argmax_op_axis1(out)
        correct_filter = (pred_label == target)
        generalized_total += x.shape[0]
        generalized_correct += correct_filter.sum().asnumpy().item()
        for i, true_label in enumerate(target.asnumpy()):
            total_label_num[int(true_label)] += 1
            if correct_filter[i].asnumpy():
                correct_label_num[int(true_label)] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    
    return personalized_correct / personalized_total, generalized_correct / generalized_total

def compute_acc(net, test_data_loader):
    net.set_train(False)
    correct, total = 0, 0
    
    for batch_idx, (x, target) in enumerate(test_data_loader):
        x = mindspore.Tensor(x, dtype=mindspore.float32)
        target = mindspore.Tensor(target, dtype=mindspore.int32)
        out = net(x)
        # 同样使用预定义的 argmax 算子
        pred_label = argmax_op_axis1(out)
        total += x.shape[0]
        correct += (pred_label == target).sum().asnumpy().item()
    
    return correct / float(total)

def evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    for net_id, _ in nets_this_round.items():
        if net_id in benign_client_list:
            val_local_dl = val_local_dls[net_id]
            data_distribution = data_distributions[net_id]

            val_acc = compute_acc(global_model, val_local_dl)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(global_model, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()
