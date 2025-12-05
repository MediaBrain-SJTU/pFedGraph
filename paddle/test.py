import numpy as np
import paddle

def compute_local_test_accuracy(model, dataloader, data_distribution):
    model.eval()
    
    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    
    # Paddle 自动处理设备，不需要手动 .cuda()
    generalized_total, generalized_correct = 0, 0
    
    # torch.no_grad() -> paddle.no_grad()
    with paddle.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            # 确保数据格式
            target = paddle.cast(target, 'int64')
            
            out = model(x)
            # torch.max -> paddle.argmax (如果只要索引)
            # 或者 paddle.topk
            pred_label = paddle.argmax(out, axis=1)
            
            correct_filter = (pred_label == target)
            generalized_total += x.shape[0]
            generalized_correct += correct_filter.sum().item()
            
            target_np = target.numpy()
            correct_np = correct_filter.numpy()
            
            for i, true_label in enumerate(target_np):
                total_label_num[true_label] += 1
                if correct_np[i]:
                    correct_label_num[true_label] += 1
                    
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    
    return personalized_correct / personalized_total, generalized_correct / generalized_total

def compute_acc(net, test_data_loader):
    net.eval()
    correct, total = 0, 0
    with paddle.no_grad():
        for batch_idx, (x, target) in enumerate(test_data_loader):
            target = paddle.cast(target, 'int64')
            out = net(x)
            pred_label = paddle.argmax(out, axis=1)
            total += x.shape[0]
            correct += (pred_label == target).sum().item()
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